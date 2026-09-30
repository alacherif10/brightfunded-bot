from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz
import os

app = Flask(__name__)

import live_scanner


def run_scanner():
    print(f"[SCHEDULER] {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')} — running scanner")
    try:
        live_scanner.main()
    except Exception as e:
        print(f"[SCHEDULER] Scanner error: {e}")


scheduler = BackgroundScheduler(timezone=pytz.UTC)
scheduler.add_job(
    run_scanner,
    'cron',
    day_of_week='mon-fri',
    hour='8-9',
    minute='0,15,30,45',
    id='london_scanner',
    replace_existing=True,
)
scheduler.start()


@app.route('/')
def health():
    return jsonify(
        status="alive",
        time=datetime.now(pytz.UTC).isoformat(),
        scheduler_running=scheduler.running,
        jobs=[str(j) for j in scheduler.get_jobs()],
    )


@app.route('/run')
def manual_run():
    try:
        run_scanner()
        return jsonify(status="ok", message="Scanner ran successfully")
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)