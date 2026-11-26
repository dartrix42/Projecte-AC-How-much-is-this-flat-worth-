"""
BASELINE SEMANA 1 · Proyecto 5 · ¿Cuánto vale este piso? (Airbnb Barcelona)

Uso (desde la raíz del repo, con el entorno activo):
    python src/baseline.py
    python src/baseline.py --data data/listings.csv.gz

Índice (cada bloque va marcado con un banner en comillas triples y una cabecera al ejecutar):
    1  CARGA
    2  LIMPIEZA                 columnas descartadas + variables derivadas
    3  DIAGNÓSTICO PRECIOS AUSENTES
    4  MERCADO / ALCANCE        turístico vs temporada (decisión 0.2)
    5  EXTREMOS                 distribución del precio y valores atípicos (decisión 0.2)
    6  FEATURES                 variables del modelo
    7  SPLIT                    aleatorio vs agrupado por host (decisión 0.3)
    8  MODELOS                  baselines + evaluación
    9  ERRORES                  dónde falla el modelo y por qué
    10 RESUMEN                  cifras listas para copiar al decision log

Regla: este script MUESTRA la evidencia; la decisión y su justificación se escriben en el README.
Unidad del precio: euros (la columna 'price' trae '$', pero las cotizaciones dicen EUR).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 45)

"""
================================================================================
 0 · CONFIGURACIÓN  (todo lo que se puede cambiar está aquí)
================================================================================
"""
SEED = 42
TEST_SIZE = 0.2
N_REPEATS = 5                 # repeticiones del split para ver el ruido de una sola partición
GROUP_COL = "host_id"

# Mercado: se define por la duración de la estancia COTIZADA (de ahí sale el precio).
SHORT_MAX_NIGHTS = 7          # cotización <= 7 noches  -> turístico
MIDTERM_MIN_NIGHTS = 31       # cotización >= 31 noches -> temporada. Entre 8 y 30 -> ambiguo (se excluye)
DROP_TOP_QUANTILE = None      # p.ej. 0.99 para quitar el 1% de precios más altos. None = no quitar nada

NUM_COLS = ["accommodates", "bedrooms", "beds", "n_baths", "shared_bath", "n_amenities", "log_host_listings"]
CAT_BARRIO = ["room_type", "neighbourhood_cleansed"]
CAT_DISTRITO = ["room_type", "neighbourhood_group_cleansed"]
MIN_FREQ_CAT = 10             # categorías con < 10 pisos se agrupan en "otras"

# Columnas a descartar (el porqué de cada grupo está en la sección 2)
URL_ID_COLS = ["listing_url", "picture_url", "host_url", "host_profile_id", "host_profile_url",
               "host_picture_url", "host_name", "last_scraped", "calendar_last_scraped"]
LEAK_COLS = ["price_quote_total_price", "price_quote_price_per_night", "price_quote_raw",
             "price_quote_checkin_date", "price_quote_checkout_date", "estimated_revenue_l365d"]
REDUNDANT_COLS = ["minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights",
                  "maximum_maximum_nights", "maximum_nights", "maximum_nights_avg_ntm",
                  "host_has_profile_pic", "host_identity_verified", "host_about", "host_location",
                  "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
                  "calculated_host_listings_count_shared_rooms", "host_listings_count",
                  "hosts_time_as_user_years", "hosts_time_as_user_months", "price", "bathrooms_text", "amenities"]

FIG_DIR, RES_DIR = Path("figures"), Path("results")
SUMMARY = {}


def banner(txt):
    print("\n" + "=" * 80 + f"\n{txt}\n" + "=" * 80)


"""
================================================================================
 1 · CARGA
================================================================================
"""
def cargar(path):
    banner("1 · CARGA")
    df = pd.read_csv(path, low_memory=False)
    print(f"Fichero: {path}  ->  {df.shape[0]:,} filas x {df.shape[1]} columnas")
    SUMMARY["filas_total"], SUMMARY["columnas_total"] = int(df.shape[0]), int(df.shape[1])
    return df


"""
================================================================================
 2 · LIMPIEZA
================================================================================
"""
def _parse_baths(s):
    """'1.5 baths' -> 1.5 ; 'Half-bath' -> 0.5 ; '0 baths' -> 0 ; nulo -> NaN"""
    if not isinstance(s, str):
        return np.nan
    s = s.lower()
    if "half" in s:
        return 0.5
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


def _n_amenities(s):
    try:
        return len(json.loads(s))
    except Exception:
        return np.nan


def limpiar(df):
    banner("2 · LIMPIEZA")
    df = df.copy()

    # 2.1 Target: '$274.00' -> 274.0 (euros). Lo que no se pueda convertir queda NaN.
    df["price_eur"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[$,€]", "", regex=True), errors="coerce")
    both = df["price_eur"].notna() & df["price_quote_price_per_night"].notna()
    same = (df.loc[both, "price_eur"] == df.loc[both, "price_quote_price_per_night"]).mean()
    print(f"price == price_quote_price_per_night donde ambos existen: {same * 100:.1f}%  "
          f"(=> el precio es la cotización de UNA estancia concreta)")

    # 2.2 Duración de la estancia cotizada (diagnóstico y definición de mercado; NO es variable del modelo)
    ci = pd.to_datetime(df["price_quote_checkin_date"], errors="coerce")
    co = pd.to_datetime(df["price_quote_checkout_date"], errors="coerce")
    df["quote_nights"] = (co - ci).dt.days

    # 2.3 Variables derivadas de columnas sucias
    df["n_baths"] = df["bathrooms_text"].apply(_parse_baths)                       # 'bathrooms' está 100% vacía
    df["shared_bath"] = df["bathrooms_text"].str.lower().str.contains("shared", na=False).astype(float)
    df.loc[df["bathrooms_text"].isna(), "shared_bath"] = np.nan
    df["n_amenities"] = df["amenities"].apply(_n_amenities)
    df["log_host_listings"] = np.log1p(df["calculated_host_listings_count"])      # tamaño del host (cola larga)

    # 2.4 Columnas a descartar, por grupos
    all_null = [c for c in df.columns if df[c].isna().all()]
    constant = [c for c in df.columns if c not in all_null and df[c].nunique(dropna=True) <= 1]
    grupos = {"100% vacías": all_null, "constantes": constant, "ids/URLs/fechas de scrape": URL_ID_COLS,
              "FUGA (derivan del precio)": LEAK_COLS, "redundantes / poca información": REDUNDANT_COLS}
    to_drop = set()
    for nombre, cols in grupos.items():
        cols = [c for c in cols if c in df.columns]
        to_drop |= set(cols)
        print(f"  - {nombre} ({len(cols)}): {', '.join(cols)}")
    df = df.drop(columns=sorted(to_drop))
    print(f"Columnas: {SUMMARY['columnas_total']} -> {df.shape[1]} "
          f"(se mantienen texto, reseñas, disponibilidad y coordenadas para semanas 2-3)")
    SUMMARY["columnas_tras_limpieza"] = int(df.shape[1])
    return df


"""
================================================================================
 3 · DIAGNÓSTICO: PRECIOS AUSENTES  (evidencia decisión 0.2)
================================================================================
"""
def diagnostico_ausentes(df):
    banner("3 · DIAGNÓSTICO · PRECIOS AUSENTES")
    miss = df["price_eur"].isna()
    print(f"Sin precio: {miss.sum():,} de {len(df):,} ({miss.mean() * 100:.1f}%)")
    SUMMARY["pct_sin_precio"] = round(float(miss.mean() * 100), 1)
    host_bucket = pd.cut(df["calculated_host_listings_count"], [0, 1, 4, 19, np.inf],
                         labels=["1", "2-4", "5-19", "20+"])
    for nombre, col in [("tipo de alojamiento", df["room_type"]), ("distrito", df["neighbourhood_group_cleansed"]),
                        ("has_availability", df["has_availability"].fillna("NA")), ("tamaño del host", host_bucket)]:
        t = miss.groupby(col, observed=True).agg(["mean", "size"])
        t["mean"] = (t["mean"] * 100).round(1)
        print(f"\n% sin precio por {nombre}:\n{t.rename(columns={'mean': '%_sin_precio', 'size': 'n'}).to_string()}")
    print("\nSi el % cambia mucho entre grupos, los ausentes NO son aleatorios: dejadlo escrito en el log.")
    return df[~miss].reset_index(drop=True)


"""
================================================================================
 4 · MERCADO / ALCANCE  (evidencia decisión 0.2)
================================================================================
"""
def definir_mercado(df):
    banner("4 · MERCADO · turístico vs temporada")
    qn = df["quote_nights"]

    # 4.1 Comparamos definiciones posibles de 'temporada'
    por_min = df["minimum_nights"] >= 31                       # los NaN cuentan como False (!)
    por_avg = df["minimum_nights_avg_ntm"] >= 31
    por_quote = qn >= 31
    print("Filas con precio:", len(df))
    print(f"  temporada por minimum_nights>=31 (NaN=False): {por_min.mean() * 100:.1f}%  "
          f"| minimum_nights nulos: {df['minimum_nights'].isna().sum():,}")
    print(f"  temporada por minimum_nights_avg_ntm>=31     : {por_avg.mean() * 100:.1f}%")
    print(f"  temporada por noches de la cotización >=31   : {por_quote.mean() * 100:.1f}%")
    mal = (por_quote & ~por_min).sum()
    print(f"  => {mal:,} pisos son de temporada por cotización pero minimum_nights los deja como turísticos")

    # 4.2 Precio según duración de la cotización: ¿dónde cambia el régimen de precios?
    bins = [0, 1, 2, 3, 7, 30, 31, 32, 400]
    bucket = pd.cut(qn, bins)
    t = df.groupby(bucket, observed=True)["price_eur"].describe(percentiles=[.25, .5, .75])[["count", "25%", "50%", "75%"]]
    print("\nPrecio (€/noche) según noches de la cotización:\n", t.round(1).to_string())
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df.loc[bucket == b, "price_eur"].dropna() for b in t.index]
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels([f"{int(b.left) + 1}-{int(b.right)}" for b in t.index])
    ax.set(yscale="log", xlabel="noches cotizadas", ylabel="€/noche (log)", title="Precio según duración de la cotización")
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG_DIR / "w1_price_by_quote_nights.png", dpi=150); plt.close(fig)

    # 4.3 Segmento (explícito, sin que los nulos se cuelen en ninguna categoría)
    df["segmento"] = np.select(
        [qn <= SHORT_MAX_NIGHTS, qn >= MIDTERM_MIN_NIGHTS, qn.notna()],
        ["turistico", "temporada", "ambiguo"], default="sin_cotizacion")
    seg = df["segmento"].value_counts()
    print(f"\nSegmentos (turístico <= {SHORT_MAX_NIGHTS} noches, temporada >= {MIDTERM_MIN_NIGHTS}):\n{seg.to_string()}")
    SUMMARY["pct_temporada_cotizacion"] = round(float(por_quote.mean() * 100), 1)
    SUMMARY["segmentos"] = {k: int(v) for k, v in seg.items()}

    # 4.4 Alcance del modelo
    scope = df[df["segmento"] == "turistico"].reset_index(drop=True)
    print(f"\nALCANCE: solo mercado turístico -> {len(scope):,} pisos "
          f"({len(scope) / len(df) * 100:.0f}% de los que tienen precio)")
    return scope


"""
================================================================================
 5 · EXTREMOS  (evidencia decisión 0.2)
================================================================================
"""
def extremos(df):
    banner("5 · EXTREMOS · distribución del precio")
    p = df["price_eur"]
    print(p.describe(percentiles=[.01, .5, .9, .95, .99]).round(1).to_string())
    print(f"\nPrecio > 1000 €: {(p > 1000).sum()}  | > 2000 €: {(p > 2000).sum()}  | > p99: {(p > p.quantile(.99)).sum()}")
    top = df.nlargest(20, "price_eur")[["id", "name", "room_type", "property_type", "accommodates",
                                        "bedrooms", "neighbourhood_cleansed", "price_eur"]]
    RES_DIR.mkdir(exist_ok=True)
    top.to_csv(RES_DIR / "w1_top_precios.csv", index=False)
    print("\n10 precios más altos (¿hoteles, grupos grandes, errores? revisadlos a mano):")
    print(top.head(10).to_string(index=False))
    print("\nTipos de alojamiento entre el 1% más caro:")
    print(df.loc[p > p.quantile(.99), "room_type"].value_counts().to_string())

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(p.clip(upper=p.quantile(.99)), bins=60); ax[0].set(title="Precio (cortado en p99)", xlabel="€/noche")
    ax[1].hist(np.log(p), bins=60); ax[1].set(title="log(precio)", xlabel="log(€/noche)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "w1_price_distribution.png", dpi=150); plt.close(fig)

    if DROP_TOP_QUANTILE:
        corte = p.quantile(DROP_TOP_QUANTILE)
        print(f"\nDROP_TOP_QUANTILE={DROP_TOP_QUANTILE}: se eliminan {(p > corte).sum()} pisos con precio > {corte:.0f} €")
        df = df[p <= corte].reset_index(drop=True)
    SUMMARY["precio_mediana_eur"] = float(p.median()); SUMMARY["precio_p99_eur"] = float(p.quantile(.99))
    SUMMARY["precio_max_eur"] = float(p.max()); SUMMARY["filas_modelo"] = int(len(df))
    return df


"""
================================================================================
 6 · FEATURES
================================================================================
"""
def revisar_features(df):
    banner("6 · FEATURES")
    cols = NUM_COLS + sorted(set(CAT_BARRIO + CAT_DISTRITO))
    t = pd.DataFrame({"%nulos": (df[cols].isna().mean() * 100).round(1), "valores_distintos": df[cols].nunique()})
    print(t.to_string())
    vc = df["neighbourhood_cleansed"].value_counts()
    print(f"\nBarrios: {len(vc)} | mediana de pisos por barrio: {vc.median():.0f} | barrios con < {MIN_FREQ_CAT} pisos: {(vc < MIN_FREQ_CAT).sum()}")
    dup = df.duplicated(["host_id", "neighbourhood_cleansed", "room_type", "accommodates", "bedrooms", "price_eur"]).sum()
    print(f"Anuncios casi idénticos (mismo host, barrio, tipo, capacidad, habitaciones y precio): {dup:,}")
    SUMMARY["anuncios_casi_identicos"] = int(dup)
    SUMMARY["hosts_distintos"] = int(df[GROUP_COL].nunique())


"""
================================================================================
 7 · MODELOS (definiciones)  -- se usan en las secciones 7 y 8
================================================================================
"""
def make_ridge(cat_cols, alpha=1.0):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_FREQ_CAT), cat_cols),
    ])
    return Pipeline([("prep", prep), ("reg", Ridge(alpha=alpha))])


def pred_mediana(train, test, keys):
    """Mediana de log(precio) por grupo, calculada SOLO con train. Si el grupo no existe, cae a tipo y luego a global."""
    tr = train.assign(lp=np.log(train["price_eur"]))
    m1 = test[keys].merge(tr.groupby(keys)["lp"].median().rename("m").reset_index(), on=keys, how="left")["m"].to_numpy()
    m2 = test[["room_type"]].merge(tr.groupby("room_type")["lp"].median().rename("m").reset_index(),
                                    on="room_type", how="left")["m"].to_numpy()
    return np.exp(np.where(np.isnan(m1), np.where(np.isnan(m2), tr["lp"].median(), m2), m1))


MODELOS = ["mediana_global", "mediana_barrio_tipo", "ridge_log_barrio", "ridge_log_distrito", "ridge_precio_barrio"]


def fit_predict(nombre, train, test):
    """Devuelve (predicciones en €, modelo_ridge_o_None)."""
    if nombre == "mediana_global":
        return np.full(len(test), np.exp(np.log(train["price_eur"]).median())), None
    if nombre == "mediana_barrio_tipo":
        return pred_mediana(train, test, ["neighbourhood_cleansed", "room_type"]), None
    cat = CAT_DISTRITO if nombre == "ridge_log_distrito" else CAT_BARRIO
    if nombre.startswith("ridge_log"):
        m = make_ridge(cat).fit(train[NUM_COLS + cat], np.log(train["price_eur"]))
        return np.exp(m.predict(test[NUM_COLS + cat])), m
    if nombre == "ridge_precio_barrio":   # alternativa SIN log: sirve para justificar la transformación
        m = make_ridge(cat).fit(train[NUM_COLS + cat], train["price_eur"])
        return m.predict(test[NUM_COLS + cat]), m
    raise ValueError(nombre)


def metricas(y, pred):
    # Evita log de negativos si el Ridge sin log predice < 0. OJO: para 'ridge_precio_barrio' esto
    # infla las métricas en log (muchas predicciones se quedan en 1 €); compararlo solo con MAE_eur.
    pred = np.clip(pred, 1.0, None)
    return {"RMSE_log": np.sqrt(mean_squared_error(np.log(y), np.log(pred))),
            "MAE_log": mean_absolute_error(np.log(y), np.log(pred)),
            "R2_log": r2_score(np.log(y), np.log(pred)),
            "MAE_eur": mean_absolute_error(y, pred), "MedAE_eur": median_absolute_error(y, pred)}


"""
================================================================================
 7 · SPLIT  (evidencia decisión 0.3): aleatorio vs agrupado por host_id
================================================================================
"""
def comparar_splits(df):
    banner("7 · SPLIT · aleatorio vs agrupado por host_id")
    splitters = {"aleatorio": ShuffleSplit(N_REPEATS, test_size=TEST_SIZE, random_state=SEED),
                 "agrupado_host": GroupShuffleSplit(N_REPEATS, test_size=TEST_SIZE, random_state=SEED)}
    rows = []
    for sname, sp in splitters.items():
        groups = df[GROUP_COL] if sname == "agrupado_host" else None   # ShuffleSplit ignora groups
        for rep, (tr, te) in enumerate(sp.split(df, groups=groups)):
            train, test = df.iloc[tr], df.iloc[te]
            comp = len(set(train[GROUP_COL]) & set(test[GROUP_COL]))
            frac = 100 * test[GROUP_COL].isin(set(train[GROUP_COL])).mean()
            for m in MODELOS:
                pred, _ = fit_predict(m, train, test)
                rows.append({"split": sname, "rep": rep, "modelo": m, "hosts_compartidos": comp,
                             "%test_con_host_en_train": frac, **metricas(test["price_eur"].to_numpy(), pred)})
    res = pd.DataFrame(rows)
    RES_DIR.mkdir(exist_ok=True)
    res.to_csv(RES_DIR / "w1_split_comparison.csv", index=False)
    cols = ["hosts_compartidos", "%test_con_host_en_train", "RMSE_log", "MAE_log", "MAE_eur", "MedAE_eur"]
    g = res.groupby(["split", "modelo"])[cols]
    out = g.mean().round(3).join(g.std().round(3).add_suffix("_std")[["RMSE_log_std", "MAE_eur_std"]])
    print(f"Media de {N_REPEATS} repeticiones (y desviación típica de RMSE_log y MAE_eur):\n{out.to_string()}")

    # Resumen de la fuga: cuánto 'mejora' el modelo con el split aleatorio
    m = "ridge_log_barrio"
    a = res[(res.split == "aleatorio") & (res.modelo == m)]["RMSE_log"].mean()
    b = res[(res.split == "agrupado_host") & (res.modelo == m)]["RMSE_log"].mean()
    print(f"\n{m}: RMSE_log aleatorio={a:.3f} vs agrupado={b:.3f}  (diferencia {b - a:+.3f})")
    SUMMARY["split_rmse_log_aleatorio"], SUMMARY["split_rmse_log_agrupado"] = round(float(a), 3), round(float(b), 3)
    SUMMARY["pct_test_con_host_en_train_aleatorio"] = round(float(res[res.split == "aleatorio"]["%test_con_host_en_train"].mean()), 1)


def split_definitivo(df):
    tr, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=SEED).split(df, groups=df[GROUP_COL]))
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    assert not set(train[GROUP_COL]) & set(test[GROUP_COL]), "¡Hay hosts compartidos entre train y test!"
    print(f"\nSPLIT DEFINITIVO (seed={SEED}): train={len(train):,}  test={len(test):,}  hosts compartidos=0")
    ids = pd.concat([train[["id"]].assign(set="train"), test[["id"]].assign(set="test")])
    ids.to_csv(RES_DIR / "w1_split_ids.csv", index=False)
    SUMMARY["train"], SUMMARY["test"] = int(len(train)), int(len(test))
    return train, test


"""
================================================================================
 8 · MODELOS · evaluación en el split definitivo
================================================================================
"""
def evaluar(train, test):
    banner("8 · MODELOS · baselines en el split definitivo (agrupado por host)")
    y = test["price_eur"].to_numpy()
    filas, preds, modelos = [], {}, {}
    for m in MODELOS:
        pred, mod = fit_predict(m, train, test)
        preds[m], modelos[m] = pred, mod
        filas.append({"modelo": m, **metricas(y, pred)})
    t = pd.DataFrame(filas).set_index("modelo").round(3)
    print(t.to_string())
    t.to_csv(RES_DIR / "w1_metrics_final.csv")
    SUMMARY["metricas_final"] = t.to_dict("index")

    # ¿Qué hace caro un piso? coeficientes del Ridge de referencia (variables estandarizadas)
    mod = modelos["ridge_log_barrio"]
    nombres = mod.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(mod.named_steps["reg"].coef_, index=nombres).sort_values()
    print("\nRidge de referencia · efecto sobre log(precio) (+ encarece, - abarata):")
    print("  Más negativos:", ", ".join(f"{k}={v:+.2f}" for k, v in coef.head(6).items()))
    print("  Más positivos:", ", ".join(f"{k}={v:+.2f}" for k, v in coef.tail(6).items()))
    return preds


"""
================================================================================
 9 · ERRORES · ¿dónde falla el modelo?
================================================================================
"""
def analizar_errores(test, preds, modelo="ridge_log_barrio"):
    banner(f"9 · ERRORES · {modelo}")
    t = test[["id", "name", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "room_type",
              "accommodates", "bedrooms", "calculated_host_listings_count", "price_eur"]].copy()
    t["pred"] = np.clip(preds[modelo], 1, None).round(1)
    t["err_eur"] = (t["pred"] - t["price_eur"]).round(1)
    t["err_pct"] = (100 * t["err_eur"] / t["price_eur"]).round(0)
    t["err_log"] = np.log(t["pred"]) - np.log(t["price_eur"])
    t.drop(columns="err_log").to_csv(RES_DIR / "w1_errors.csv", index=False)

    show = ["id", "name", "neighbourhood_cleansed", "room_type", "accommodates", "bedrooms", "price_eur", "pred", "err_pct"]
    print("5 mayores SOBREpredicciones (predice mucho más de lo real):")
    print(t.nlargest(5, "err_eur")[show].to_string(index=False))
    print("\n5 mayores SUBpredicciones (el precio real es mucho mayor):")
    print(t.nsmallest(5, "err_eur")[show].to_string(index=False))

    # Error por subgrupo: ¿falla más en algún tipo de entrada?
    def tabla(col, nombre):
        g = t.groupby(col, observed=True).agg(n=("err_log", "size"), sesgo_log=("err_log", "mean"),
                                              MAE_log=("err_log", lambda s: s.abs().mean()),
                                              MAE_eur=("err_eur", lambda s: s.abs().mean())).round(3)
        print(f"\nError por {nombre} (sesgo_log>0: sobrepredice, <0: infrapredice):\n{g.to_string()}")
    tabla("room_type", "tipo de alojamiento")
    tabla("neighbourhood_group_cleansed", "distrito")
    tabla(pd.qcut(t["price_eur"], 5, duplicates="drop"), "quintil de precio real")
    tabla(pd.cut(t["calculated_host_listings_count"], [0, 1, 4, 19, np.inf], labels=["1", "2-4", "5-19", "20+"]),
          "tamaño del host")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(np.log(t["price_eur"]), np.log(t["pred"]), s=6, alpha=.35)
    lim = [np.log(t["price_eur"]).min(), np.log(t["price_eur"]).max()]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlabel="log(precio real)", ylabel="log(precio predicho)", title=f"{modelo} · test")
    fig.tight_layout(); fig.savefig(FIG_DIR / "w1_pred_vs_real.png", dpi=150); plt.close(fig)


"""
================================================================================
 10 · RESUMEN · cifras para el decision log
================================================================================
"""
def resumen():
    banner("10 · RESUMEN · cifras para copiar al README (decision log)")
    for k, v in SUMMARY.items():
        if k != "metricas_final":
            print(f"  {k}: {v}")
    (RES_DIR / "w1_summary.json").write_text(json.dumps(SUMMARY, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {RES_DIR}/ (métricas, errores, ids del split, resumen) y {FIG_DIR}/ (gráficos)")


"""
================================================================================
 MAIN · orden de ejecución
================================================================================
"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/2026-09-17/listings.csv.gz")
    args = ap.parse_args()

    df = cargar(args.data)                  # 1
    df = limpiar(df)                        # 2
    df = diagnostico_ausentes(df)           # 3  (devuelve solo filas con precio)
    df = definir_mercado(df)                # 4  (devuelve solo el mercado turístico)
    df = extremos(df)                       # 5
    revisar_features(df)                    # 6
    comparar_splits(df)                     # 7
    train, test = split_definitivo(df)      # 7
    preds = evaluar(train, test)            # 8
    analizar_errores(test, preds)           # 9
    resumen()                               # 10


if __name__ == "__main__":
    main()
