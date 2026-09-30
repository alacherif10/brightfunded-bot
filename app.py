from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz
import live_scanner  # Your existing scanner script

app = Flask(__name__)

def run_scanner():
    """Wrapper to run the scanner and log output."""
    print(f"[{datetime.now()}] Running London session scanner...")
    try:
        live_scanner.main()
    except Exception as e:
        print(f"Scanner error: {e}")

# Configure scheduler (London time = UTC in winter, UTC+1 in summer)
# Adjust the hours based on the current season.
# Winter (Nov-Mar): 09:00-10:59 UTC  -> hour='9-10'
# Summer (Apr-Oct): 08:00-09:59 UTC  -> hour='8-9'
scheduler = BackgroundScheduler(timezone=pytz.UTC)
scheduler.add_job(
    run_scanner,
    'cron',
    day_of_week='mon-fri',
    hour='9-10',      # <-- CHANGE TO '8-9' DURING SUMMER (Apr-Oct)
    minute='0,15,30,45'
)
scheduler.start()

@app.route('/')
def health():
    """Health check endpoint for UptimeRobot."""
    return jsonify(status="alive", time=datetime.now().isoformat())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)