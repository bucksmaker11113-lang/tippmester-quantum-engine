# backend/main.py

from backend.pipeline.master_pipeline import MasterPipeline
import json

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def main():
    config = load_config()
    pipeline = MasterPipeline(config)
    result = pipeline.run_daily()

    print("\n=== DAILY TIPPMESTER REPORT ===")
    print("Dátum:", result["date"])
    print("Tippek száma:", len(result["tips"]))

    if "kombi" in result:
        print("Kombi:", result["kombi"])

if __name__ == "__main__":
    main()
