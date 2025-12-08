import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import skew
from scipy.stats import kurtosis

def descargar_tickers(tickers, carpeta='MarketData', start='2000-01-01', end=None):
    """
    Descarga datos históricos de una lista de tickers usando yfinance
    y guarda un archivo CSV por cada ticker que tenga datos.

    Parámetros:
    - tickers: lista de símbolos (ej. ['AAPL', 'MSFT', '^GSPC'])
    - carpeta: carpeta donde se guardarán los CSV
    - start: fecha de inicio (YYYY-MM-DD)
    - end: fecha de fin (YYYY-MM-DD, opcional)
    """
    # Si no se especifica fecha final, usa la fecha actual
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    # Crear carpeta si no existe
    os.makedirs(carpeta, exist_ok=True)

    for tic in tickers:
        print(f"Descargando datos de {tic}...")
        try:
            data = yf.download(tic, start=start, end=end, progress=False)

            # Si no hay datos, saltar
            if data.empty:
                print(f"⚠️ No se encontraron datos para {tic}, se omite.")
                continue

            # Dejar solo las columnas necesarias
            df = data.reset_index()[['Date', 'Close']]

            # Guardar archivo CSV
            ruta = os.path.join(carpeta, f"{tic}.csv")
            df.to_csv(ruta, index=False)

        except Exception as e:
            print(f"Error descargando {tic}: {e}")
            continue



def daily_return(ticker, data_dir="MarketData"):
    """
    Carga una serie temporal desde un archivo CSV y calcula los rendimientos diarios.

    Parámetros
    ----------
    ticker : str
        Símbolo del activo (por ejemplo: 'AAPL').
    data_dir : str, opcional
        Directorio donde se encuentran los archivos CSV. Por defecto 'MarketData'.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas: ['date', 'close', 'return']
    """

    # Ruta del archivo
    file_path = os.path.join(os.getcwd(), data_dir, f"{ticker}.csv")

    # Leer las columnas necesarias
    df = pd.read_csv(
        file_path,
        usecols=["Date", "Close"],
        parse_dates=["Date"]
    )

    # Limpiar y preparar
    df = (
        df.sort_values("Date")
          .rename(columns={"Date": "date", "Close": "close"})
    )

    # Convertir a numérico (por si hay texto como 'N/A')
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Eliminar filas donde 'close' sea NaN antes del cálculo
    df = df.dropna(subset=["close"])

    # Calcular rendimientos diarios
    df["return"] = df["close"].pct_change()

    # Eliminar la primera fila (NaN en return)
    df = df.dropna(subset=["return"]).reset_index(drop=True)

    return df

def sync_timeseries(tickers, data_dir="MarketData"):
    """
    Carga y sincroniza series temporales de retornos diarios para varios tickers.

    Parámetros
    ----------
    tickers : list[str]
        Lista de símbolos (por ejemplo: ['XLK', 'XLF', 'XLV']).
    data_dir : str, opcional
        Directorio donde se encuentran los archivos CSV. Por defecto 'MarketData'.

    Retorna
    -------
    df : pd.DataFrame
        DataFrame con las fechas y los retornos sincronizados de cada ticker.
    mtx_var_covar : np.ndarray
        Matriz de varianza-covarianza.
    mtx_correl : np.ndarray
        Matriz de correlaciones.
    """

    # Cargar y preparar todas las series de retornos
    all_returns = []

    for ticker in tickers:
        t = daily_return(ticker, data_dir=data_dir)
        t = t[['date', 'return']].rename(columns={'return': ticker})
        all_returns.append(t)

    # Unir todas las series por la columna 'date' (intersección automática)
    df = all_returns[0]
    for t in all_returns[1:]:
        df = pd.merge(df, t, on='date', how='inner')

    # Limpiar y ordenar
    df = df.dropna().sort_values('date').reset_index(drop=True)

    # Calcular matrices
    returns_only = df.drop(columns='date')
    mtx_var_covar = returns_only.cov().values
    mtx_correl = returns_only.corr().values

    # Mostrar resultados
    print("Primeras filas del DataFrame sincronizado:")
    print(df.head(), "\n")

    print("Matriz Varianza-Covarianza:")
    print(mtx_var_covar, "\n")

    print("Matriz de Correlaciones:")
    print(mtx_correl, "\n")

    return df, mtx_var_covar, mtx_correl

def beta(port, benchmark):
    cov = np.cov(port, benchmark, ddof=0)[0,1]
    var = np.var(benchmark)
    return cov / var

def media(r):
    return r.mean()

def volatilidad(r):
    return r.std()

def sharpe(r, rf=0.0):
    excess = r - rf/252
    return np.sqrt(252) * excess.mean() / excess.std()

def sortino(r, rf=0.0):
    excess = r - rf/252
    downside = excess[excess < 0].std()
    return np.sqrt(252) * excess.mean() / downside

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def var_95(r):
    return np.percentile(r, 5)

def cvar_95(r):
    v = var_95(r)
    return r[r <= v].mean()

def sesgo(r):
    return skew(r)

def curtosis(r):
    return kurtosis(r)

def construir_portafolio(data, pesos):
    retornos_df = data.pct_change().dropna()
    pesos = np.array(pesos, dtype=float)
    pesos = pesos / pesos.sum()
    portafolio = (retornos_df * pesos).sum(axis=1)
    return portafolio

# Función para calcular métricas (aquí simularemos con valores aleatorios)
def calcular_metricas_portfolio(portafolio, benchmark):
    # AQUÍ DEBES IMPLEMENTAR TUS CÁLCULOS REALES
    metricas = {
        'rendimiento': 10000,
        'volatilidad': portafolio.std(),
        'beta': beta(portafolio, benchmark),
        'sharpe': sharpe(portafolio, rf=0.05),
        'sortino': sortino(portafolio, rf=0.05),
        'max_drawdown': max_drawdown(portafolio),
        'var': var_95(portafolio),
        'cvar': cvar_95(portafolio),
        'sesgo': sesgo(portafolio),
        'curtosis': curtosis(portafolio),
        'rend_positivos': 10000,
        'delta_benchmark': 10000
    }
    return metricas