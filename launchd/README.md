# Local auto-restart

This folder contains macOS `launchd` LaunchAgents for:

- `generate_shared_candles_stocks.py`
- `tradeBot_main.py`
- `tradeBot_main_7.py`

Each source plist is rendered by `scripts/install_launch_agents.sh` with the
current project directory before it is installed into `~/Library/LaunchAgents`.
Each agent runs through the project virtualenv and `/usr/bin/caffeinate -is`.
That gives two local protections:

- `KeepAlive` restarts the process if it exits or crashes.
- `caffeinate -is` prevents automatic idle sleep while the process is running.

It cannot keep the bot running while the Mac is fully shut down, manually put to
sleep, or sleeping because a laptop lid is closed.

Install/start:

```bash
./scripts/install_launch_agents.sh
```

Status:

```bash
./scripts/status_launch_agents.sh
```

Stop/remove:

```bash
./scripts/uninstall_launch_agents.sh
```

Logs are written to:

```text
logs/launchd/
```
