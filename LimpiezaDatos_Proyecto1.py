import os
import glob
import pandas as pd

# =========================================================
# CONFIGURACIÓN
# =========================================================
CARPETA = os.path.expanduser("~/Downloads/Datos Ecobici")
CARPETA_SALIDA = os.path.join(CARPETA, "parquet_procesados")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

CHUNK_SIZE = 300_000  # puedes subir a 500_000 si tu PC aguanta

COLUMNAS_FINALES = [
    "Genero_Usuario",
    "Edad_Usuario",
    "Bici",
    "Ciclo_Estacion_Retiro",
    "Fecha_Retiro",
    "Hora_Retiro",
    "Ciclo_EstacionArribo",
    "Fecha_Arribo",
    "Hora_Arribo",
]

MAPEO_COLUMNAS = {
    "genero_usuario": "Genero_Usuario",
    "genero usuario": "Genero_Usuario",
    "edad_usuario": "Edad_Usuario",
    "edad usuario": "Edad_Usuario",
    "bici": "Bici",
    "ciclo_estacion_retiro": "Ciclo_Estacion_Retiro",
    "ciclo estacion retiro": "Ciclo_Estacion_Retiro",
    "ciclo_estacionarribo": "Ciclo_EstacionArribo",
    "ciclo estacionarribo": "Ciclo_EstacionArribo",
    "ciclo_estacion_arribo": "Ciclo_EstacionArribo",
    "ciclo estacion arribo": "Ciclo_EstacionArribo",
    "fecha_retiro": "Fecha_Retiro",
    "fecha retiro": "Fecha_Retiro",
    "hora_retiro": "Hora_Retiro",
    "hora retiro": "Hora_Retiro",
    "fecha_arribo": "Fecha_Arribo",
    "fecha arribo": "Fecha_Arribo",
    "hora_arribo": "Hora_Arribo",
    "hora arribo": "Hora_Arribo",
}

# =========================================================
# FUNCIONES
# =========================================================
def normalizar_nombre(col: str) -> str:
    col = str(col).strip().lower()
    col = " ".join(col.split())
    col = col.replace("-", "_")
    return MAPEO_COLUMNAS.get(col, col)

def detectar_delimitador(path_csv: str) -> str:
    with open(path_csv, "r", encoding="utf-8", errors="ignore") as f:
        primera = f.readline()
    return ";" if primera.count(";") > primera.count(",") else ","

def limpiar_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # Normalizar nombres de columnas
    df.columns = [normalizar_nombre(c) for c in df.columns]

    # Crear faltantes si no existen
    for col in COLUMNAS_FINALES:
        if col not in df.columns:
            df[col] = pd.NA

    # Quedarnos solo con las columnas importantes
    df = df[COLUMNAS_FINALES].copy()

    # Limpiar strings
    for col in COLUMNAS_FINALES:
        df[col] = df[col].astype("string").str.strip()

    # Reemplazar valores nulos raros
    df = df.replace({
        "NULL": pd.NA,
        '"NULL"': pd.NA,
        "null": pd.NA,
        "": pd.NA,
    })

    # Edad como número
    df["Edad_Usuario"] = pd.to_numeric(df["Edad_Usuario"], errors="coerce").astype("Float32")

    # Fechas y horas
    retiro_txt = df["Fecha_Retiro"].fillna("") + " " + df["Hora_Retiro"].fillna("")
    arribo_txt = df["Fecha_Arribo"].fillna("") + " " + df["Hora_Arribo"].fillna("")

    df["FechaHora_Retiro"] = pd.to_datetime(
        retiro_txt,
        dayfirst=True,
        errors="coerce"
    )
    df["FechaHora_Arribo"] = pd.to_datetime(
        arribo_txt,
        dayfirst=True,
        errors="coerce"
    )

    # Variables derivadas
    df["Anio"] = df["FechaHora_Retiro"].dt.year.astype("Float32")
    df["Mes"] = df["FechaHora_Retiro"].dt.month.astype("Float32")
    df["DiaSemana"] = df["FechaHora_Retiro"].dt.dayofweek.astype("Float32")  # 0=lunes, 6=domingo
    df["Hora"] = df["FechaHora_Retiro"].dt.hour.astype("Float32")
    df["Es_FinDeSemana"] = df["DiaSemana"].isin([5, 6])

    # Duración en minutos
    df["Duracion_Minutos"] = (
        (df["FechaHora_Arribo"] - df["FechaHora_Retiro"]).dt.total_seconds() / 60
    ).astype("Float32")

    # Filtrar duraciones absurdas
    df = df[
        df["Duracion_Minutos"].isna() |
        ((df["Duracion_Minutos"] >= 0) & (df["Duracion_Minutos"] <= 300))
    ].copy()

    return df

# =========================================================
# PROCESAMIENTO
# =========================================================
archivos = sorted(glob.glob(os.path.join(CARPETA, "*.csv")))

if not archivos:
    raise FileNotFoundError(f"No se encontraron CSV en: {CARPETA}")

print(f"Se encontraron {len(archivos)} archivos CSV.\n")

resumen = []
errores = []

for archivo in archivos:
    nombre = os.path.basename(archivo)
    nombre_sin_ext = os.path.splitext(nombre)[0]
    salida_parquet = os.path.join(CARPETA_SALIDA, f"{nombre_sin_ext}.parquet")

    print(f"Procesando: {nombre}")

    try:
        sep = detectar_delimitador(archivo)
        partes = []
        total_filas = 0

        for chunk in pd.read_csv(
            archivo,
            sep=sep,
            dtype=str,              # <- TODO como texto, aquí está la clave
            chunksize=CHUNK_SIZE,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False
        ):
            limpio = limpiar_chunk(chunk)
            partes.append(limpio)
            total_filas += len(limpio)
            print(f"  Chunk procesado: {len(limpio):,} filas limpias")

        if not partes:
            raise ValueError("No se pudo procesar ningún chunk.")

        df_final = pd.concat(partes, ignore_index=True)

        # Guardar parquet
        df_final.to_parquet(salida_parquet, index=False, engine="pyarrow", compression="snappy")

        resumen.append((nombre, len(df_final)))
        print(f"  Guardado: {salida_parquet}")
        print(f"  Filas finales: {len(df_final):,}\n")

        del partes
        del df_final

    except Exception as e:
        errores.append((nombre, str(e)))
        print(f"  Error en {nombre}: {e}\n")

# =========================================================
# RESUMEN
# =========================================================
print("\nProceso terminado.\n")
print("Archivos procesados correctamente:")
for nombre, filas in resumen:
    print(f"- {nombre}: {filas:,} filas")

if errores:
    print("\nArchivos con error:")
    for nombre, err in errores:
        print(f"- {nombre}: {err}")

print(f"\nParquets generados en:\n{CARPETA_SALIDA}")