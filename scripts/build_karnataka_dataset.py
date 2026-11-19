"""Karnataka-only hourly forecast and verification data builder."""
import argparse, asyncio, io, json, zipfile
from datetime import date, timedelta
from pathlib import Path
import httpx, pandas as pd
GEONAMES="https://download.geonames.org/export/dump/cities500.zip"
ADMIN1="https://download.geonames.org/export/dump/admin1CodesASCII.txt"
HIST="https://historical-forecast-api.open-meteo.com/v1/forecast"
PREV="https://previous-runs-api.open-meteo.com/v1/forecast"
OBS="https://archive-api.open-meteo.com/v1/archive"
FORECAST=["temperature_2m","relative_humidity_2m","precipitation","precipitation_probability","wind_speed_10m"]
BASE=["temperature_2m","relative_humidity_2m","precipitation","wind_speed_10m"]
COLS=["geonameid","name","asciiname","alternatenames","latitude","longitude","feature_class","feature_code","country_code","cc2","admin1_code","admin2","admin3","admin4","population","elevation","dem","timezone","modification_date"]
def load_locations(limit=30):
    with httpx.Client(timeout=120,follow_redirects=True) as c:
        zr=c.get(GEONAMES); zr.raise_for_status()
        ar=c.get(ADMIN1); ar.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(zr.content)) as z:
        member=next(n for n in z.namelist() if n.endswith(".txt"))
        cities=pd.read_csv(z.open(member),sep="\\t",header=None,names=COLS,dtype={"country_code":str,"admin1_code":str})
    admin=pd.read_csv(io.BytesIO(ar.content),sep="\\t",header=None,names=["code","name","asciiname"])
    state=admin[admin.asciiname.str.casefold().eq("karnataka")]
    if state.empty: raise RuntimeError("Karnataka not found in GeoNames admin1 catalogue.")
    code=state.iloc[0].code.split(".",1)[1]
    cities=cities[(cities.country_code=="IN")&(cities.admin1_code.astype(str)==code)&(cities.feature_class=="P")].copy()
    cities["population"]=pd.to_numeric(cities.population,errors="coerce").fillna(0)
    cities=cities.sort_values("population",ascending=False).drop_duplicates(["name","latitude","longitude"]).head(limit)
    if len(cities)<limit: raise RuntimeError(f"Only {len(cities)} Karnataka populated places found; expected {limit}.")
    cities["location_id"]=cities.geonameid.astype(str)
    cities["elevation_m"]=pd.to_numeric(cities.dem,errors="coerce").fillna(0)
    return cities[["location_id","name","latitude","longitude","elevation_m"]].reset_index(drop=True)
async def get_json(client,url,params):
    last=None
    for attempt in range(5):
        try:
            r=await client.get(url,params=params)
            if r.status_code==429 or r.status_code>=500:
                last=f"HTTP {r.status_code}: {r.text[:300]}"; await asyncio.sleep(min(45,2**(attempt+2))); continue
            r.raise_for_status(); data=r.json()
            if data.get("error"): raise RuntimeError(data.get("reason","Open-Meteo API error"))
            return data
        except (httpx.TimeoutException,httpx.TransportError) as e:
            last=repr(e); await asyncio.sleep(min(45,2**(attempt+2)))
    raise RuntimeError(f"Retries exhausted: {last}")
def to_frame(payload,prefix,variables):
    h=payload.get("hourly",{})
    if not h.get("time"): raise RuntimeError(f"No hourly timestamps for {prefix}")
    d={"valid_time":pd.to_datetime(h["time"],utc=True)}
    for v in variables:
        if v not in h: raise RuntimeError(f"Missing API variable {v} in {prefix}")
        d[f"{prefix}_{v}"]=h[v]
    return pd.DataFrame(d)
async def one_location(client,loc,start,end):
    common={"latitude":float(loc.latitude),"longitude":float(loc.longitude),"start_date":start,"end_date":end,"timezone":"UTC"}
    pvars=[f"{v}_previous_day{lead}" for lead in (1,2,3) for v in BASE]
    hist,prev,obs=await asyncio.gather(
        get_json(client,HIST,{**common,"hourly":",".join(FORECAST)}),
        get_json(client,PREV,{**common,"hourly":",".join(pvars)}),
        get_json(client,OBS,{**common,"hourly":",".join(BASE)}))
    frame=to_frame(hist,"historical_forecast",FORECAST)
    frame=frame.merge(to_frame(prev,"previous_run",pvars),on="valid_time",how="inner")
    frame=frame.merge(to_frame(obs,"observed",BASE),on="valid_time",how="inner")
    metadata={"location_id":str(loc.location_id),"location":loc["name"],"country_code":"IN","state":"Karnataka","latitude":float(loc.latitude),"longitude":float(loc.longitude),"elevation_m":float(loc.elevation_m)}
    for key,value in reversed(list(metadata.items())): frame.insert(0,key,value)
    return frame
async def build(locations,start,end,out):
    path=Path(out); path.parent.mkdir(parents=True,exist_ok=True); frames=[]
    async with httpx.AsyncClient(timeout=150,follow_redirects=True) as client:
        for i,loc in locations.iterrows():
            print(f"[{i+1}/{len(locations)}] {loc['name']}")
            f=await one_location(client,loc,start,end); frames.append(f)
            pd.concat(frames,ignore_index=True).to_parquet(path,index=False)
            print(f"  saved {sum(len(x) for x in frames):,} rows")
            await asyncio.sleep(.5)
    data=pd.concat(frames,ignore_index=True)
    if data.location_id.nunique()!=len(locations) or data.empty: raise RuntimeError("Incomplete Karnataka dataset.")
    report={"state":"Karnataka","locations":int(data.location_id.nunique()),"rows":int(len(data)),"start_utc":str(data.valid_time.min()),"end_utc":str(data.valid_time.max()),"forecast_sources":["Historical Forecast API","Previous Runs API"],"verification_source":"Historical Weather API","lead_times_days":[1,2,3],"missing_fraction":data.isna().mean().to_dict()}
    Path("artifacts").mkdir(exist_ok=True); Path("artifacts/karnataka_dataset_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--days",type=int,default=365); p.add_argument("--end-date",default=(date.today()-timedelta(days=6)).isoformat()); p.add_argument("--locations",type=int,default=30); p.add_argument("--output",default="data/karnataka_forecast_training.parquet"); a=p.parse_args()
    if a.days<30: raise ValueError("--days must be at least 30")
    end=date.fromisoformat(a.end_date); start=end-timedelta(days=a.days-1)
    asyncio.run(build(load_locations(a.locations),start.isoformat(),end.isoformat(),a.output))
if __name__=="__main__": main()
