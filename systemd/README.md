# systemd or service in RHEL/Rocky
- Ref: https://medium.com/@charlessampa/introduction-to-systemd-in-linux-a-beginners-guide-29cbdef42ad5

- Why systemd?
  - Parallel service startup
  - Service dependency management
  - Integrated logging via journald
  - Service monitoring & auto-restart
  - Resource control: limit cpu/memory usage per service
- Typical locations
  - `/etc/systemd/system/`
  - `/lib/systemd/system` or `/usr/lib/systemd/system`
- Demo
  - A python web server: `python3 myserver.py`
  - A sample service file at `/etc/systemd/system/myservice.service`
```
[Unit]
Description=Simple Python Web Server
After=network.target
[Service]
ExecStart=/usr/bin/python3 /home/yourusername/myserver.py
WorkingDirectory=/home/yourusername
Restart=always
User=yourusername
[Install]
WantedBy=multi-user.target
```
  - `sudo systemctl daemon-reload`
  - `sudo systemctl start myserver`
  - `sudo systemctl enable myserver`
  - `sudo systemstl status myserver`
  - To view logs, `sudo journalctl -u myserver`

