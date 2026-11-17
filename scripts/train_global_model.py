import argparse,json
from ml.train import train
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",default="data/global_weather_1y.csv");a=p.parse_args()
    result=train(a.data,"artifacts")
    print(json.dumps(result,indent=2))
