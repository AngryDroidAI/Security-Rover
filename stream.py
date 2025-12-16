systemd/rover.service:[Unit]
Description=Security Rover Core Controller
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rover
ExecStart=/usr/bin/python3 services/rover_controller.py
Restart=always

[Install]
WantedBy=multi-user.target

systemd/rover-video.service:[Unit]
Description=Security Rover Video Stream
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rover
ExecStart=/usr/bin/python3 services/video_stream.py
Restart=always

[Install]
WantedBy=multi-user.target


systemd/rover-dashboard.service (if you keep dashboard standalone):[Unit]
Description=Security Rover Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rover
ExecStart=/usr/bin/python3 web/dashboard_server.py
Restart=always

[Install]
WantedBy=multi-user.target

[Unit]
Description=Security Rover Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rover
ExecStart=/usr/bin/python3 web/dashboard_server.py
Restart=always

[Install]
WantedBy=multi-user.target

sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rover.service rover-video.service rover-dashboard.service
sudo systemctl start rover.service rover-video.service rover-dashboard.service
