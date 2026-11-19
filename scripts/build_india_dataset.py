"""India-wide hourly forecast and verification dataset builder."""
import argparse, asyncio, io, json, zipfile
from datetime import date, timedelta
from pathlib import Path
import httpx, pandas as pd

GEONAMES="https://download.geonames.org/export/dump/cities500.zip"
HIST="https://historical-forecast-api.open-meteo.com/v1/forecast"
PREV="https://previous-runs-api.open-meteo.com/v1/forecast"
OBS="https://archive-api.open-meteo.com/v1/archive"
FORECAST=["temperature_2m","relative_humidity_2m","precipitation","precipitation_probability","wind_speed_10m"]
BASE=["temperature_2m","relative_humidity_2m","precipitation","wind_speed_10m"]
COLS=["geonameid","name","asciiname","alternatenames","latitude","longitude","feature_class","feature_code","country_code","cc2","admin1_code","admin2","admin3","admin4","population","elevation","dem","timezone","modification_date"]

def load_locations(limit=30):
    with httpx.Client(timeout=120,follow_redirects=True) as client:
        response=client.get(GEONAMES); response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        member=next(name for name in archive.namelist() if name.endswith(".txt"))
        cities=pd.read_csv(archive.open(member),sep="\t",header=None,names=COLS,
                           dtype={"country_code":str,"admin1_code":str})
    cities=cities[(cities.country_code=="IN")&(cities.feature_class=="P")].copy()
    cities["population"]=pd.to_numeric(cities.population,errors="coerce").fillna(0)
    cities=cities.sort_values("population",ascending=False).drop_duplicates(
        ["name","latitude","longitude"]).head(limit)
    if len(cities)<limit:
        raise RuntimeError(f"Only {len(cities)} Indian populated places found; expected {limit}.")
    cities["location_id"]=cities.geonameid.astype(str)
    cities["elevation_m"]=pd.to_numeric(cities.dem,errors="coerce").fillna(0)
    cities["admin1_code"]=cities.admin1_code.fillna("").astype(str)
    return cities[["location_id","name","latitude","longitude","elevation_m","admin1_code"]].reset_index(drop=True)

async def get_json(client,url,params):
    last=None
    for attempt in range(5):
        try:
            response=await client.get(url,params=params)
            if response.status_code==429 or response.status_code>=500:
                last=f"HTTP {response.status_code}: {response.text[:300]}"
                await asyncio.sleep(min(45,2**(attempt+2))); continue
            response.raise_for_status(); payload=response.json()
            if payload.get("error"): raise RuntimeError(payload.get("reason","Open-Meteo API error"))
            return payload
        except (httpx.TimeoutException,httpx.TransportError) as exc:
            last=repr(exc); await asyncio.sleep(min(45,2**(attempt+2)))
    raise RuntimeError(f"Retries exhausted: {last}")

def to_frame(payload,prefix,variables):
    hourly=payload.get("hourly",{})
    if not hourly.get("time"): raise RuntimeError(f"No hourly timestamps for {prefix}")
    data={"valid_time":pd.to_datetime(hourly["time"],utc=True)}
    for variable in variables:
        if variable not in hourly: raise RuntimeError(f"Missing API variable {variable} in {prefix}")
        data[f"{prefix}_{variable}"]=hourly[variable]
    return pd.DataFrame(data)

async def one_location(client,location,start,end):
    common={"latitude":float(location.latitude),"longitude":float(location.longitude),
            "start_date":start,"end_date":end,"timezone":"UTC"}
    previous=[f"{variable}_previous_day{lead}" for lead in (1,2,3) for variable in BASE]
    historical,previous_runs,observations=await asyncio.gather(
        get_json(client,HIST,{**common,"hourly":",".join(FORECAST)}),
        get_json(client,PREV,{**common,"hourly":",".join(previous)}),
        get_json(client,OBS,{**common,"hourly":",".join(BASE)}))
    frame=to_frame(historical,"historical_forecast",FORECAST)
    frame=frame.merge(to_frame(previous_runs,"previous_run",previous),on="valid_time",how="inner")
    frame=frame.merge(to_frame(observations,"observed",BASE),on="valid_time",how="inner")
    metadata={"location_id":str(location.location_id),"location":location["name"],
              "country_code":"IN","admin1_code":str(location.admin1_code),
              "latitude":float(location.latitude),"longitude":float(location.longitude),
              "elevation_m":float(location.elevation_m)}
    for key,value in reversed(list(metadata.items())): frame.insert(0,key,value)
    return frame

async def build(locations,start,end,output):
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); frames=[]
    async with httpx.AsyncClient(timeout=150,follow_redirects=True) as client:
        for index,location in locations.iterrows():
            print(f"[{index+1}/{len(locations)}] {location['name']}")
            frame=await one_location(client,location,start,end); frames.append(frame)
            pd.concat(frames,ignore_index=True).to_parquet(path,index=False)
            print(f"  saved {sum(len(part) for part in frames):,} rows")
            await asyncio.sleep(.5)
    data=pd.concat(frames,ignore_index=True)
    if data.location_id.nunique()!=len(locations) or data.empty:
        raise RuntimeError("Incomplete India dataset.")
    report={"country":"India","locations":int(data.location_id.nunique()),"rows":int(len(data)),
            "start_utc":str(data.valid_time.min()),"end_utc":str(data.valid_time.max()),
            "forecast_sources":["Historical Forecast API","Previous Runs API"],
            "verification_source":"Historical Weather API","lead_times_days":[1,2,3],
            "missing_fraction":data.isna().mean().to_dict()}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/india_dataset_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--days",type=int,default=365)
    parser.add_argument("--end-date",default=(date.today()-timedelta(days=6)).isoformat())
    parser.add_argument("--locations",type=int,default=30)
    parser.add_argument("--output",default="data/india_forecast_training.parquet")
    args=parser.parse_args()
    if args.days<30: raise ValueError("--days must be at least 30")
    end=date.fromisoformat(args.end_date); start=end-timedelta(days=args.days-1)
    asyncio.run(build(load_locations(args.locations),start.isoformat(),end.isoformat(),args.output))

if __name__=="__main__": main()
