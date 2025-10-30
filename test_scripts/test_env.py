from solana_trading_bot_bundle.common.constants import APP_NAME, local_appdata_dir, appdata_dir, logs_dir, data_dir, config_path, env_path, db_path, token_cache_path, ensure_app_dirs, prefer_appdata_file
import os
from dotenv import load_dotenv

env_path = "C:/Users/Admin/Desktop/Solana_Trading_Bot_Installer_Ready/.env"
if not os.path.exists(env_path):
    raise FileNotFoundError(f".env file not found at {env_path}")
load_dotenv(dotenv_path=env_path)

print("Loaded .env file from:", env_path)
print("SOLANA_PRIVATE_KEY from os.getenv:", os.getenv("SOLANA_PRIVATE_KEY"))
print("SOLANA_PRIVATE_KEY from os.environ:", os.environ.get("SOLANA_PRIVATE_KEY"))




