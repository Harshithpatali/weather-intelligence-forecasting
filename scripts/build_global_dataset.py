"""Build 1-year global training data: 30 populated places x 40 countries.
Open-Meteo ERA5 archive reanalysis, batched multi-coordinate requests.
"""
import argparse,asyncio,io,zipfile
from datetime import date,timedelta
from pathlib import Path
import httpx,pandas as pd

COUNTRIES=["US","CA","MX","BR","AR","CL","GB","IE","FR","DE","ES","IT","NL","BE","CH","AT","SE","NO","FI","PL","CZ","PT","GR","TR","RU","IN","PK","BD","LK","NP","CN","JP","KR","TH","VN","ID","MY","SG","AU","NZ"]
GEONAMES="https://download.geonames.org/export/dump/cities15000.zip"
ARCHIVE="https://archive-api.open-meteo.com/v1/archive"

def load_locations(path):
    p=Path(path)
    if p.exists(): cities=pd.read_csv(p)
    else:
        print("Downloading GeoNames cities15000 catalogue...")
        with httpx.Client(timeout=120,follow_redirects=True) as client:
            response=client.get(GEONAMES);response.raise_for_status()
        z=zipfile.ZipFile(io.BytesIO(response.content));name=next(n for n in z.namelist() if n.endswith(".txt"))
        cols=["geonameid","name","asciiname","alternatenames","latitude","longitude","feature_class","feature_code","country_code","cc2","admin1","admin2","admin3","admin4","population","elevation","dem","timezone","modification_date"]
        cities=pd.read_csv(z.open(name),sep="\t",header=None,names=cols,dtype={"country_code":str},low_memory=False)
        cities=cities[(cities.feature_class=="P") & cities.country_code.isin(COUNTRIES)].copy()
        cities["population"]=pd.to_numeric(cities.population,errors="coerce").fillna(0)
        cities=cities.sort_values(["country_code","population"],ascending=[True,False]).groupby("country_code").head(30)
        cities.to_csv(p,index=False)
    counts=cities.groupby("country_code").size()
    missing=[c for c in COUNTRIES if counts.get(c,0)<30]
    if missing: raise ValueError(f"GeoNames catalogue lacks 30 populated places for: {missing}. Supply a complete data/locations.csv with 30 places per country.")
    cities=cities[cities.country_code.isin(COUNTRIES)].sort_values(["country_code","population"],ascending=[True,False]).groupby("country_code").head(30)
    return cities.reset_index(drop=True)

async def fetch_batch(client,locs,start,end):
    params={"latitude":",".join(str(x) for x in locs.latitude),
      "longitude":",".join(str(x) for x in locs.longitude),"start_date":start,"end_date":end,
      "daily":"temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
      "hourly":"relative_humidity_2m","timezone":"UTC","wind_speed_unit":"ms"}
    for attempt in range(5):
        try:
            r=await client.get(ARCHIVE,params=params)
            if r.status_code in (429,500,502,503): await asyncio.sleep(2**attempt);continue
            r.raise_for_status()
            data=r.json()
            return data if isinstance(data,list) else [data]
        except (httpx.HTTPError,ValueError):
            if attempt==4: raise
            await asyncio.sleep(2**attempt)
    raise RuntimeError("Archive API retries exhausted")

async def build(locations,start,end,out):
    records=[]
    async with httpx.AsyncClient(timeout=180,follow_redirects=True) as client:
        for offset in range(0,len(locations),10):
            batch=locations.iloc[offset:offset+10]
            results=await fetch_batch(client,batch,start,end)
            if len(results)!=len(batch): raise RuntimeError("Multi-coordinate response count mismatch.")
            for (_,loc),item in zip(batch.iterrows(),results):
                daily=item.get("daily",{});hourly=item.get("hourly",{})
                h=pd.DataFrame({"time":hourly.get("time",[]),"humidity":hourly.get("relative_humidity_2m",[])})
                if not h.empty:
                    h["date"]=pd.to_datetime(h.time).dt.date.astype(str)
                    humidity=h.groupby("date").humidity.mean().to_dict()
                else: humidity={}
                for i,day in enumerate(daily.get("time",[])):
                    records.append({"location_id":str(loc.get("geonameid",loc.get("location_id"))),
                      "city":loc["name"],"country_code":loc["country_code"],"latitude":float(loc["latitude"]),
                      "longitude":float(loc["longitude"]),"elevation_m":float(loc.get("dem",loc.get("elevation",0)) or 0),
                      "date":day,"temperature_c":daily["temperature_2m_mean"][i],
                      "humidity_pct":humidity.get(day),"rainfall_mm":daily["precipitation_sum"][i],
                      "wind_speed_ms":daily["wind_speed_10m_max"][i]})
            print(f"Collected {min(offset+len(batch),len(locations))}/{len(locations)} locations")
            await asyncio.sleep(.2)
    frame=pd.DataFrame(records).dropna()
    Path(out).parent.mkdir(parents=True,exist_ok=True);frame.to_csv(out,index=False)
    print(f"Saved {len(frame):,} rows, {frame.location_id.nunique()} locations, {frame.country_code.nunique()} countries")

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
