"""
Script para descargar datos históricos de ETFs y guardarlos en CSV
Ejecuta este script ANTES de correr la aplicación Streamlit

Instala yfinance primero: pip install yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ETFs de la estrategia 1: REGIONES
tickers_regiones = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]

# ETFs de la estrategia 2: SECTORES DE ESTADOS UNIDOS
tickers_sectores = ["XLC", "XLY", "XLP", "XLE", "XLF",
                    "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]

print("=" * 70)
print("DESCARGANDO DATOS HISTÓRICOS DE ETFs")
print("=" * 70)

print("\n📊 Descargando datos de ETFs de REGIONES...")
print(f"Tickers: {', '.join(tickers_regiones)}")

# Descargar precios ajustados de los últimos 4 años
data_regiones = yf.download(tickers_regiones, period="4y", progress=True)["Close"]

# Verificar si hay columnas con datos faltantes
print("\n🔍 Verificando calidad de datos REGIONES:")
for ticker in tickers_regiones:
    if ticker in data_regiones.columns:
        null_count = data_regiones[ticker].isnull().sum()
        total_count = len(data_regiones)
        pct_null = (null_count / total_count) * 100
        
        if pct_null > 50:
            print(f"   ⚠️  {ticker}: {pct_null:.1f}% datos faltantes ({null_count}/{total_count})")
        elif pct_null > 0:
            print(f"   ⚡ {ticker}: {pct_null:.1f}% datos faltantes ({null_count}/{total_count}) - Interpolando...")
            # Interpolar valores faltantes
            data_regiones[ticker] = data_regiones[ticker].interpolate(method='linear')
        else:
            print(f"   ✅ {ticker}: OK")
    else:
        print(f"   ❌ {ticker}: NO DESCARGADO")

# Si EWC tiene muchos problemas, intentar descargarlo individualmente
if 'EWC' in data_regiones.columns and data_regiones['EWC'].isnull().sum() > len(data_regiones) * 0.5:
    print("\n🔄 Intentando descargar EWC individualmente con período más largo...")
    try:
        ewc_data = yf.download("EWC", period="5y", progress=False)["Close"]
        # Alinear con las fechas de data_regiones
        data_regiones['EWC'] = ewc_data.reindex(data_regiones.index).interpolate(method='linear')
        print(f"   ✅ EWC descargado: {data_regiones['EWC'].isnull().sum()} nulls restantes")
    except Exception as e:
        print(f"   ❌ Error descargando EWC: {e}")

print("\n📊 Descargando datos de ETFs de SECTORES...")
print(f"Tickers: {', '.join(tickers_sectores)}")

data_sectores = yf.download(tickers_sectores, period="4y", progress=True)["Close"]

# Verificar calidad de datos de sectores
print("\n🔍 Verificando calidad de datos SECTORES:")
for ticker in tickers_sectores:
    if ticker in data_sectores.columns:
        null_count = data_sectores[ticker].isnull().sum()
        total_count = len(data_sectores)
        pct_null = (null_count / total_count) * 100
        
        if pct_null > 50:
            print(f"   ⚠️  {ticker}: {pct_null:.1f}% datos faltantes ({null_count}/{total_count})")
        elif pct_null > 0:
            print(f"   ⚡ {ticker}: {pct_null:.1f}% datos faltantes - Interpolando...")
            data_sectores[ticker] = data_sectores[ticker].interpolate(method='linear')
        else:
            print(f"   ✅ {ticker}: OK")
    else:
        print(f"   ❌ {ticker}: NO DESCARGADO")

# Eliminar filas donde TODOS los valores son NaN
data_regiones = data_regiones.dropna(how='all')
data_sectores = data_sectores.dropna(how='all')

# Forward fill para cualquier NaN restante al inicio
data_regiones = data_regiones.fillna(method='ffill').fillna(method='bfill')
data_sectores = data_sectores.fillna(method='ffill').fillna(method='bfill')

# Guardar en CSV
print("\n💾 Guardando archivos CSV...")
data_regiones.to_csv('data_regiones.csv')
data_sectores.to_csv('data_sectores.csv')

print("\n" + "=" * 70)
print("✅ ARCHIVOS GENERADOS EXITOSAMENTE")
print("=" * 70)
print(f"\n📁 data_regiones.csv")
print(f"   - Filas: {len(data_regiones)} días")
print(f"   - Columnas: {len(data_regiones.columns)} ETFs")
print(f"   - Rango: {data_regiones.index.min().date()} → {data_regiones.index.max().date()}")
print(f"   - Tickers: {', '.join(data_regiones.columns)}")

print(f"\n📁 data_sectores.csv")
print(f"   - Filas: {len(data_sectores)} días")
print(f"   - Columnas: {len(data_sectores.columns)} ETFs")
print(f"   - Rango: {data_sectores.index.min().date()} → {data_sectores.index.max().date()}")
print(f"   - Tickers: {', '.join(data_sectores.columns)}")

# Verificación final de nulls
print("\n🔍 VERIFICACIÓN FINAL DE DATOS FALTANTES:")
nulls_regiones = data_regiones.isnull().sum().sum()
nulls_sectores = data_sectores.isnull().sum().sum()

if nulls_regiones == 0 and nulls_sectores == 0:
    print("   ✅ No hay datos faltantes en ningún archivo")
else:
    if nulls_regiones > 0:
        print(f"   ⚠️  data_regiones.csv tiene {nulls_regiones} valores faltantes")
    if nulls_sectores > 0:
        print(f"   ⚠️  data_sectores.csv tiene {nulls_sectores} valores faltantes")

print("\n🚀 Ahora puedes ejecutar la aplicación Streamlit:")
print("   streamlit run app.py")
print("=" * 70)