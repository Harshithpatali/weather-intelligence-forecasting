"""Build one-year global weather training data with resumable, resilient downloads."""
import argparse,asyncio,io,zipfile
from datetime import date,timedelta
from pathlib import Path
import httpx,pandas as pd

COUNTRIES=["US","CA","MX","BR","AR","CL","GB","IE","FR","DE","ES","IT","NL","BE","CH","AT","SE","NO","FI","PL","CZ","PT","GR","TR","RU","IN","PK","BD","LK","NP","CN","JP","KR","TH","VN","ID","MY","SG","AU","NZ"]
GEONAMES="https://download.geonames.org/export/dump/cities500.zip"
ARCHIVE="https://archive-api.open-meteo.com/v1/archive"

def load_locations(path):
    p=Path(path)
    if p.exists():
        cities=pd.read_csv(p)
    else:
        print("Downloading GeoNames cities500 catalogue...")
        with httpx.Client(timeout=120,follow_redirects=True) as client:
            response=client.get(GEONAMES);response.raise_for_status()
        z=zipfile.ZipFile(io.BytesIO(response.content));name=next(n for n in z.namelist() if n.endswith(".txt"))
        cols=["geonameid","name","asciiname","alternatenames","latitude","longitude","feature_class","feature_code","country_code","cc2","admin1","admin2","admin3","admin4","population","elevation","dem","timezone","modification_date"]
        cities=pd.read_csv(z.open(name),sep="\\t",header=None,names=cols,dtype={"country_code":str},low_memory=False)
        cities=cities[(cities.feature_class=="P") & cities.country_code.isin(COUNTRIES)].copy()
        cities["population"]=pd.to_numeric(cities.population,errors="coerce").fillna(0)
        cities=cities.sort_values(["country_code","population"],ascending=[True,False]).groupby("country_code").head(30)
        p.parent.mkdir(parents=True,exist_ok=True);cities.to_csv(p,index=False)
    cities["population"]=pd.to_numeric(cities.get("population",0),errors="coerce").fillna(0)
    counts=cities.groupby("country_code").size()
    missing=[c for c in COUNTRIES if counts.get(c,0)<30]
    if missing: raise ValueError(f"GeoNames catalogue lacks 30 places for: {missing}")
    return cities[cities.country_code.isin(COUNTRIES)].sort_values(["country_code","population"],ascending=[True,False]).groupby("country_code").head(30).reset_index(drop=True)

def params_for(locs,start,end):
    return {"latitude":",".join(str(x) for x in locs.latitude),
      "longitude":",".join(str(x) for x in locs.longitude),"start_date":start,"end_date":end,
      "daily":"temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
      "hourly":"relative_humidity_2m","timezone":"UTC","wind_speed_unit":"ms"}

async def request_archive(client,locs,start,end):
    params=params_for(locs,start,end)
    last_error=None
    for attempt in range(6):
        try:
            response=await client.get(ARCHIVE,params=params)
            if response.status_code==429 or response.status_code>=500:
                retry=response.headers.get("Retry-After")
                wait=float(retry) if retry and retry.replace(".","",1).isdigit() else min(60,3*(2**attempt))
                last_error=f"HTTP {response.status_code}: {response.text[:500]}"
                print(f"Archive temporary error ({last_error}); retrying in {wait:.0f}s")
                await asyncio.sleep(wait);continue
            if response.is_error:
                raise RuntimeError(f"Open-Meteo archive HTTP {response.status_code}: {response.text[:1000]}")
            payload=response.json()
            results=payload if isinstance(payload,list) else [payload]
            if len(results)!=len(locs):
                raise RuntimeError(f"Archive returned {len(results)} location payloads for {len(locs)} requested.")
            for item in results:
                if isinstance(item,dict) and item.get("error"):
                    raise RuntimeError(f"Open-Meteo archive error: {item.get('reason','unknown error')}")
            return results
        except (httpx.TimeoutException,httpx.TransportError) as exc:
            last_error=repr(exc);await asyncio.sleep(min(60,3*(2**attempt)))
    raise RuntimeError(f"Archive request retries exhausted. Last error: {last_error}")

async def fetch_resilient(client,batch,start,end):
    try:
        return await request_archive(client,batch,start,end)
    except Exception as batch_error:
        if len(batch)==1:
            raise RuntimeError(f"Failed location {batch.iloc[0]['name']} ({batch.iloc[0]['country_code']}): {batch_error}") from batch_error
        print(f"Batch of {len(batch)} failed ({batch_error}); falling back to single-location requests.")
        results=[]
        for _,loc in batch.iterrows():
            results.extend(await request_archive(client,batch.loc[[loc.name]],start,end))
            await asyncio.sleep(1.0)
        return results

def rows_from_payload(loc,item):
    daily=item.get("daily",{});hourly=item.get("hourly",{})
    h=pd.DataFrame({"time":hourly.get("time",[]),"humidity":hourly.get("relative_humidity_2m",[])})
    humidity={}
    if not h.empty:
        h["date"]=pd.to_datetime(h.time).dt.date.astype(str)
        humidity=h.groupby("date").humidity.mean().to_dict()
    rows=[]
    for i,day in enumerate(daily.get("time",[])):
        rows.append({"location_id":str(loc.get("geonameid",loc.get("location_id"))),
          "city":loc["name"],"country_code":loc["country_code"],"latitude":float(loc["latitude"]),
          "longitude":float(loc["longitude"]),"elevation_m":float(loc.get("dem",loc.get("elevation",0)) or 0),
          "date":day,"temperature_c":daily["temperature_2m_mean"][i],
          "humidity_pct":humidity.get(day),"rainfall_mm":daily["precipitation_sum"][i],
          "wind_speed_ms":daily["wind_speed_10m_max"][i]})
    return rows

async def build(locations,start,end,out):
    output=Path(out);output.parent.mkdir(parents=True,exist_ok=True)
    completed=set()
    records=[]
    if output.exists():
        existing=pd.read_csv(output)
        if not existing.empty:
            records=existing.to_dict("records")
            completed=set(existing.location_id.astype(str).unique())
            print(f"Resuming: {len(completed)} locations already saved.")
    pending=locations[~locations.geonameid.astype(str).isin(completed)] if "geonameid" in locations else locations[~locations.location_id.astype(str).isin(completed)]
    async with httpx.AsyncClient(timeout=180,follow_redirects=True) as client:
        for offset in range(0,len(pending),10):
            batch=pending.iloc[offset:offset+10]
            results=await fetch_resilient(client,batch,start,end)
            if len(results)!=len(batch): raise RuntimeError("Response count did not match requested locations.")
            for (_,loc),item in zip(batch.iterrows(),results):
                records.extend(rows_from_payload(loc,item))
            pd.DataFrame(records).to_csv(output,index=False)
            print(f"Saved progress: {min(offset+len(batch),len(pending))}/{len(pending)} new locations; {len(records):,} rows total.")
            await asyncio.sleep(1.0)
    frame=pd.DataFrame(records)
    if frame.empty: raise RuntimeError("No historical weather records were returned.")
    frame=frame.dropna(subset=["temperature_c","humidity_pct","rainfall_mm","wind_speed_ms"])
    frame.to_csv(output,index=False)
    countries=frame.country_code.nunique();places=frame.location_id.nunique()
    print(f"Saved {len(frame):,} daily rows for {places} locations in {countries} countries to {output}")
    if places<1200 or countries<40: raise RuntimeError(f"Incomplete dataset: {places}/1200 places, {countries}/40 countries.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--locations",default="data/locations.csv")
    p.add_argument("--output",default="data/global_weather_1y.csv")
    p.add_argument("--end-date",default=(date.today()-timedelta(days=6)).isoformat())
    p.add_argument("--days",type=int,default=365)
    a=p.parse_args()
    loc=load_locations(a.locations);end=date.fromisoformat(a.end_date)
    asyncio.run(build(loc,(end-timedelta(days=a.days-1)).isoformat(),end.isoformat(),a.output))
if __name__=="__main__":main()
