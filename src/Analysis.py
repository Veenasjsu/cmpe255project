import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from google.cloud import bigquery
import dash
#from jupyter_dash import JupyterDash
from dash import dcc, html
from dash.dependencies import Input, Output
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import DBSCAN
import plotly.graph_objs as go
import torch
import torch.nn as nn
from dash import Dash, dcc, html, Input, Output

def load_data(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    df = client.query(f"SELECT * FROM `{table_ref}`").to_dataframe()
    df['Date'] = pd.to_datetime(df['Date'])
    #print(df.head())
    return df

def preprocess(df, seasonal=True):
    df_f = df[df['Seasonally Adjusted'] == seasonal]
    ts = df_f.pivot_table(index='Date', columns='Industry Title', values='Current Employment').sort_index()
    ts = ts.asfreq('MS')
    ts = ts.interpolate(method='time').ffill().bfill()
    scaler = MinMaxScaler()
    ts = pd.DataFrame(scaler.fit_transform(ts), index=ts.index, columns=ts.columns)

    return ts, scaler


class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # take the output of the last time step
        out = self.linear(out[:, -1, :])
        return out

def forecast_lstm_pytorch(series, periods=12, look_back=12, epochs=50, lr=0.001):
    # Scale the series to [0,1]
    values = series.values.astype('float32').reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()

    # Prepare sequences
    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i + look_back])
        y.append(scaled[i + look_back])
    X = torch.tensor(np.array(X)).unsqueeze(-1)  # shape [samples, look_back, 1]
    y = torch.tensor(np.array(y)).unsqueeze(-1)  # shape [samples, 1]

    # Initialize model, loss, optimizer
    model = LSTMForecast(input_size=1, hidden_size=50)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Forecast future points
    model.eval()
    preds = []
    last_seq = torch.tensor(scaled[-look_back:]).unsqueeze(0).unsqueeze(-1)  # shape [1, look_back, 1]
    with torch.no_grad():
        for _ in range(periods):
            pred = model(last_seq)
            preds.append(pred.item())
            next_seq = last_seq.squeeze().numpy().flatten().tolist()[1:] + [pred.item()]
            last_seq = torch.tensor(next_seq).unsqueeze(0).unsqueeze(-1)

    # Invert scaling
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # Build forecast index
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(), periods=periods, freq='ME')
    return idx, preds


def forecast_arima(series, periods=12, auto=False):
    order, seasonal_order = (1,1,1), (1,1,1,12)
    if auto:
        best_aic, best_cfg = np.inf, None
        for p in range(2):
            for d in range(2):
                for q in range(2):
                    try:
                        res = ARIMA(series, order=(p,d,q), seasonal_order=seasonal_order).fit()
                        if res.aic < best_aic:
                            best_aic, best_cfg = res.aic, (p,d,q)
                    except:
                        continue
        if best_cfg:
            order = best_cfg
    model = ARIMA(series, order=order, seasonal_order=seasonal_order).fit()
    fc = model.forecast(steps=periods)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(), periods=periods, freq='ME')
    return idx, fc

def forecast_prophet(series, periods=12):
    dfp = series.reset_index().rename(columns={'Date':'ds', series.name:'y'})
    m = Prophet()
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq='ME')
    f = m.predict(future)
    return f['ds'][-periods:].values, f['yhat'][-periods:].values


def generate_report(ts, scaler, industries, report_path='report.md'):
    lines = [f"# Employment Trend Analysis Report", f"Generated on {pd.Timestamp.now()}", ""]
    for ind in industries:
        series = ts[ind]
        idx_a, fa = forecast_arima(series, auto=True)
        mae = mean_absolute_error(series[-len(fa):], fa)
        rmse = np.sqrt(mean_squared_error(series[-len(fa):], fa))
        
        lines.append(f"## {ind}")
        lines.append(f"- ARIMA MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        plt.figure()
        plt.plot(series.index, series, label='Historical')
        plt.plot(idx_a, fa, label='Forecast')
        
        plt.legend()
        img = f"{ind.replace(' ','_')}.png"
        plt.savefig(img)
        plt.close()
        lines.append(f"![{ind}]({img})")
        lines.append("")
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Report written to {report_path}")


def create_dashboard(ts, industries):

    def make_figures(ind):
        series = ts[ind].dropna()

        # ARIMA forecast
        idx_a, fa = forecast_arima(series, auto=True)
        # Prophet forecast
        idx_p, fp = forecast_prophet(series)
        # PyTorch LSTM forecast
        idx_l, fl = forecast_lstm_pytorch(series)

        # Build a Plotly Figure
        fig = go.Figure()

        # Historical trace
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode='lines', name='Historical'
        ))

        # ARIMA trace
        fig.add_trace(go.Scatter(
            x=idx_a, y=fa,
            mode='lines', name='ARIMA'
        ))

        # Prophet trace
        fig.add_trace(go.Scatter(
            x=idx_p, y=fp,
            mode='lines', name='Prophet'
        ))

        # LSTM trace
        fig.add_trace(go.Scatter(
            x=idx_l, y=fl,
            mode='lines', name='LSTM'
        ))

        # **Here’s the title addition:**
        fig.update_layout(
            title=f"Forecast Comparison for {ind}",
            xaxis_title="Date",
            yaxis_title="Current Employment",
            height=400,
            template="simple_white"
        )

        return fig

    #app = JupyterDash(__name__)

    app = Dash(
        __name__,
        assets_folder=os.path.join(os.path.dirname(__file__), '..', 'assets')
    )

    default_ind = industries[0]
    init_fig = make_figures(default_ind)

    app.layout = html.Div([
        html.H1("Employment Trend Analysis"),
        dcc.Dropdown(
            id='industry-dropdown',
            options=[{'label':i,'value':i} for i in industries],
            value=default_ind,
            clearable=False
        ),
        dcc.Graph(id='forecast-graph', figure=init_fig)
    ])

    @app.callback(
        Output('forecast-graph', 'figure'),
        Input('industry-dropdown', 'value')
    )
    def update_forecast(ind):
        return make_figures(ind)

    return app


def compute_all_metrics(ts, periods=12):
    records = []
    for ind in ts.columns:
        series = ts[ind]
        # 1. Get forecasts
        idx_ar, fc_ar = forecast_arima(series, periods=periods)
        _,        fc_pr = forecast_prophet(series.rename(ind), periods=periods)
        _,        fc_ls = forecast_lstm_pytorch(series, periods=periods)

        # Align true values; here we compare on the last <periods> points
        true = series[-periods:]

        # 2. MAPE helper
        def mape(y_true, y_pred):
            return (np.abs((y_true - y_pred) / y_true).dropna().mean()) * 100

        # 3. Record metrics
        for model, pred in [('ARIMA', fc_ar), ('Prophet', fc_pr), ('LSTM', fc_ls)]:
            mae   = mean_absolute_error(true, pred)
            mse   = mean_squared_error( true, pred )       
            rmse  = np.sqrt(mse)                           
            mapev = mape(true, pred)
            records.append({
                'industry':  ind,
                'model':     model,
                'MAE':       mae,
                'RMSE':      rmse,
                'MAPE(%)':   mapev
            })

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv('metrics_summary.csv', index=False)

    # Display a pivoted table for slides
    # display(df_metrics.pivot_table(
    #     index='industry',
    #     columns='model',
    #     values=['MAE','RMSE','MAPE(%)']
    # ))
    return df_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true')
    args, _ = parser.parse_known_args()
    pid = os.getenv('PROJECT_ID','dm-project-458907')
    did = os.getenv('DATASET_ID','CurrentEmploymentStatistics')
    tid = os.getenv('TABLE_ID','ces')
    df = load_data(pid, did, tid)
    ts, scaler = preprocess(df)
    inds = ts.columns.tolist()
    #metrics_df = compute_all_metrics(ts)
    print('args.........................',args)
    if args.report:
      generate_report(ts, scaler, inds)
    else:
      app = create_dashboard(ts, inds)
      #app.run(mode='inline', port=8052)
      app.run(debug=False, host='127.0.0.1', port=8053)


if __name__ == '__main__':
    main()
