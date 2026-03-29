# Observability with Grafana, Prometheus, Loki, Alloy and Tempo
- Instructor: Aref Karimi
## Section 1: Introduction

### 1. Introduction
- https://github.com/aussiearef/Prometheus
- https://github.com/aussiearef/grafana-udemy

## Section 2: Foundations of Observability

### 2. From Monoliths to Microservices: Why We Need Observability
- Monolithic architecture
  - All services in one application
  - User interfance and business logic in one application
  - All services shareed one database
  - To make a change, the entire application is deployed
- Microservcie architecture
  - Individual services
  - EAch service has its own storage
  - UI and servcies are separate
  - Changes are deployed without deploying the entire SW
- Foundations of observability
  - Many services to monitor
  - Intra-service communications can fail
  - More vulnerable to security threats
  - More changes are deployed
    
### 3. What is Monitoring
- 3 questions of monitoring
  - Is the service on?
  - Is the service functioning as expected?
  - Is the service performing well?
- Telemetry data: data collected for monitoring
- Metrics used to measure the DevOps success
  - Mean time to detection (MTTD)
  - Mean time to resolve (MTTR)

### 4. Methods of Monitoring
- Layers of systems
  - UI layer: Core web vitals
  - Service layer: RED method
  - Infrastructure layer: USE method
- RED Method (Request Oriented)
  - Rate (throughput): request per second
  - Errors: failed requests i.e, HTTP 500
  - Duration: latency or transaction response time
- USE method (Resource oriented)
  - Utilization : CPU usage %
  - Saturation: Network queue length. Zero = Good
  - Errors: i.e., disk write error. Zero = Good
- Four Golden Signals Method (RED+S)
  - Latency
  - Traffic (throughput)
  - Errors
  - Saturation (Resources at 100 capacity)
- Core web vitals
  - Largest contentful paint (perceived page load)
  - First input delay (perceived responsiveness)
  - Cumulative layout shift (perceived stability)

### 5. What is Observability
- What to monitor?

### 6. Methods of Collecting Metrics. Push vs. Scrape
- Push method: sends the metrics to an endpoint via TCP, UDP or HTTP
  - Ex: sending metrics to StatsD, to be stored in Graphite
- Scrape method: applications and microservices provide APIs for the time series database, to read the metrics
  - Ex: Prometheus scraping metrics
- How to decide:
  - Types of systems and applications
  - Scalability
  - Complexity

### 7. Types of Telemetry Data
- Types of telemetry data (MELT)
  - Metric: an aggregated value representing events ina period of time
  - Event: an action happened at a given time
  - Log: a very detailed representation of an event
  - Trace: shows the interactions of microservices to fulfill a request


### Role Play 1: Let's Talk About Foundations of Observability
- Response is very slow

## Section 3: Installing Prometheus & Collecting Metrics on Any OS

### 8. Installing Prometheus on Windows

### 9. Installing Prometheus on Mac OS

### 10. Installing Prometheus on Linux (Ubuntu)
- Prometheus: https://prometheus.io/download/
- sudo groupadd --system prometheus
- sudo useradd -s /sbin/nologin --system -g prometheus prometheus
- sudo mkdir /var/lib/prometheus
- sudo mkdir -p /etc/prometheus/rules
- sudo mkdir -p /etc/prometheus/rules.s
- sudo mkdir -p /etc/prometheus/files_sd
- tar zxf prometheus-3.10.0.linux-amd64.tar.gz 
- cd prometheus-3.10.0.linux-amd64/
- sudo mv prometheus promtool /usr/local/bin/
- prometheus --version
- sudo mv prometheus.yml /etc/prometheus/
- sudo vi /etc/systemd/system/prometheus.service
```sh
[Unit]
Description=Prometheus
Documentation=https://prometheus.io/docs/introduction/overview/
Wants=network-online.target
After=network-online.target
[Service]
Type=simple
User=prometheus
Group=prometheus
ExecReload=/bin/kill -HUP $MAINPID
ExecStart=/usr/local/bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/var/lib/prometheus \
  --web.console.templates=/etc/prometheus/consoles \
  --web.console.libraries=/etc/prometheus/console_libraries \
  --web.listen-address=0.0.0.0:9090 \
  --web.external-url=
SyslogIdentifier=prometheus
Restart=always
[Install]
WantedBy=multi-user.target
```
- sudo chown -R prometheus:prometheus /etc/prometheus
- sudo chwon -R prometheus:prometheus /etc/prometheus/*
- sudo chmod -R 775 /etc/prometheus
- sudo chmod -R 775 /etc/prometheus/*
- sudo chwon -R prometheus:prometheus /var/lib/prometheus
- sudo chwon -R prometheus:prometheus /var/lib/prometheus/*
- sudo systemctl daemon-reload
- sudo systemctl start prometheus
- sudo systemctl enable prometheus
- systemctl status prometheus
- Open browser with http://localhost:9090

### 11. Collecting Metrics (Unix , Linux and Mac)
- Collects data from SQL server or cloud service through scheduled job to avoid traffic
  - NOT scalable
- Let Prometheus connect each SQL server/cloud service through exporter -> Scraping
  - Requires Push Gateway

### 12. Node Exporter - Part 1 (Linux, Mac)
- Node: every UNIX-based kernel & computer

### 13. Node Exporter - Part 2 (Linux, Mac)
- Node exporter listens on port 9100/TCP
- At client node:
  - `sudo apt-get update`
  - Download node exporter: https://prometheus.io/download/#node_exporter
  - `tar zxf node_exporter-1.10.2.linux-amd64.tar.gz`
  - `cd node_exporter-1.10.2.linux-amd64/`
  - `./node_exporter`
  - Open browser and find http://localhost:9100
![node_exporter](./ch13_node_exp.png)

### 14. Node Exporter - Part 3 (Linux, Mac)
- Edit /etc/prometheus/prometheus.yml in the prometheus server
  - job_name, targets IP
- sudo systemctl restart prometheus

### 15. Running Node Exporter as a Service on Ubuntu
- In the client side
- sudo groupadd --system prometheus
- sudo useradd -s /sbin/nologin --system -g prometheus prometheus
- sudo mkdir /var/lib/node/
- sudo mv node_exporter /var/lib/node/
- sudo vi /etc/systemd/system/node.service
```bash
[Unit]
Description=Prometheus Node Exporter
Documentation=https://prometheus.io/docs/introduction/overview/
Wants=network-online.target
After=network-online.target
[Service]
Type=simple
User=prometheus
Group=prometheus
ExecReload=/bin/kill -HUP $MAINPID
ExecStart=/var/lib/node/node_exporter
SyslogIdentifier=prometheus_node_exporter
Restart=always
[Install]
WantedBy=multi-user.target
```
- sudo chown -R prometheus:prometheus /var/lib/node
- sudo chmod -R 775 /var/lib/node
- sudo systemctl daemon-reload
- sudo systemctl start node
- sudo systemctl enable node
- systemctl status node
- To see the log file: `journalctl -u node`
- When Prometheus cannot query machine metrics:
  - Check :9100/metrics vs :9090/metrics
  - localhost:9100 lists machine metrics like `node_cpu_seconds_total`
  - If you have prometheus and node in the same machine, you need to add a following segment into `prometheus.yml`
```yml
  - job_name: "node"
    static_configs:
      - targets: ["localhost:9100"]
        labels:
          app: "node"
```
  - sudo systemctl daemon-reload
  - sudo systemctl restart node
  - sudo systemctl restart prometheus.service

### 16. Data Model of Prometheus
- Prometheus stores data as time series
- Every time series is identified by metric name and labels
- Labels are a key and value pair
- Labels are optional
  - `<metric name> {key=value, key=value,...}`
  - Ex: `auth_api_hit {count=1, time_taken=800}`

### 17. Data Types in Prometheus
- In PromQL
  - Scalar: Float, String
  - Instant vectors: selection of a set of time series and a single sample value for each at a given timestamp (instant)
    - Only a metric name is specified
    - Ex: `auth_api_hit 5`, `auth_api_hit {count=1, time_taken=800} 1`
  - Range vectors: Similar to instant vectors but they select a range of samples
    - label_name[time_spec]
    - Ex: `auth_api_hit[5m]`
- Units:
  - ms: milliseconds
  - s: seconds
  - m: minutes
  - h: hours
  - d: days
  - w: weeks
  - y: years
- Scalar data query:
![scalar](./ch17_scalar.png)
- Range vector data query:
![vector](./ch17_vector.png)
- Adjust `scrape_interval` in prometheus.yml

### 18. Binary Arithmatic Operators in Prometheus
- +,-,*,/,%,^
- Scalar + Instant Vector: applies to every element of instant vector
- Instant Vector + Instant Vector: Only common keys (after summed) are returned

### 19. Binary Comparison Operators in Prometheus
- ==, !=, >, <,
- 1 == 1 returns 1
- 1 == 2 returns 0
- When Instant vector is compared, only corresponding (calculated as true) elements are returned

### 20. Set Binary Operators in Prometheus
- and, or, unless

### 21. Matchers and Selectors in Prometheus
- `<metric name> {filter_key=value, filter_key=value, ...}`
- `=`: two values must be equal
- `!=`: two values must NOT be equal
- `=~`: value on left must match the Regular Expression (regex) on right
  - Ex: `prometheus_http_requests_total{code=~"2.*",job="prometheus"}`
- `!~`: value on left must NOT match the Regular Expression (regex) on right
![regex_ex](./ch21_regex.png)

### 22. Aggregation Operators
- sum, min, max, avg, count
- group: Group elements. All values in resulting vector are equal to 1
- count_values: counts the number of elements with the same values
- topk: largest elements by sample value
- bottomk: smallest elements by sample value
- stddev, stdvar

### 23. Time Offsets
- `prometheus_http_requests_total offset 10m`
![ch23_offset](./ch23_offset.png)
- Note that time zone in Grafana/Prometheus is UTC by default

### 24. Clamping and Checking Functions
- `absent(<Instant Vector>)`: checks if an instant vector has any members. Returns an empty vector if parameter has elements
  - Returns the requested vector WHEN IT DOES NOT EXIST
- `absent_over_time(<range Vector>)`
- `abs(<Instant Vector>)`: converts all values to positive numbers
- `ceil(<Instant Vector>)`
- `floor(<Instant Vector>)`
- `clamp(<Instant Vector>, min, max)`
- `clamp_min(<Instant Vector>, min)`
- `clamp_max(<Instant Vector>, max)`

### 25. Delta and iDelta
- `day_of_month(<Instant Vector>)`
- `day_of_week(<Instant Vector>)`
- `delta(<Instant Vector>)`: can only be used with Gauges
  - Ex: `delta(node_cpu_temp[2h])`
- `idelta(<Range Vector>)`: returns the difference b/w first and last items 

### 26. Sorting and TimeStamp
- `log2(<Instant Vector>)`: returns binary logarithm of each scalar value
- `log10(<Instant Vector>)`: returns decimal logarithm of each scalar value
- `ln(<Instant Vector>)`: returns neutral logarithm of each scalar value
- `sort(<Instant Vector>)`: sorts elements in ascending order
- `sort_desc(<Instant Vector>)`: sorts elements in descending order
- time(): recent time stamp

### 27. Aggregations Over Time
- `avg_over_time(<range Vector>)`: returns the average of items in a range vector
- `sum_over_time(<range Vector>)`
- `min_over_time(<range Vector>)`
- `max_over_time(<range Vector>)`
- `count_over_time(<range Vector>)`

### 28. (Optional) Collecting Metrics in Mac using Node Exporter

### 29. (Optional) Collecting Metrics in Windows using MMI Exporter

### Role Play 2: Let's chat about Prometheus

## Section 4: Installing and Configuring Grafana

### 30. Cloud or On-Premises?
- Grafana cloud: a fully managed cloud-hosted platform

### 31. Installing Grafana on Ubuntu
- https://grafana.com/grafana/download
- sudo apt-get install -y adduser libfontconfig1 musl
- wget https://dl.grafana.com/grafana-enterprise/release/12.4.1/grafana-enterprise_12.4.1_22846628243_linux_amd64.deb
- sudo dpkg -i grafana-enterprise_12.4.1_22846628243_linux_amd64.deb
- sudo systemctl daemon-reload
- sudo systemctl enable grafana-server
- sudo systemctl start grafana-server
- systemctl status grafana-server
- Open `http://localhost:3000`
- Login as admin/admin
  - Will ask to change passwd

### 32. Installing Grafana on Amazon Linux, Red Hat, CentOS, RHEL, and Fedora

### 33. Installing Grafana on Windows

### 34. Installing Grafana on Mac with Homebrew

### 35. Configuring Grafana
- cd /etc/grafana
- sudo cp grafana.ini custom.ini
- sudo vi grafana.ini
  - Remove semicolon in the first column to activate the line
  - logs = /var/log/grafana
  - type = sqlite3
- sudo systemctl restart grafana-server

### 36. Launching Grafana Stack and Prometheus with Docker
- https://github.com/aussiearef/grafana-udemy
- Install docker then reboot
- https://github.com/aussiearef/grafana-udemy/blob/main/docker/docker-compose.yaml

## Section 5: Using Grafana

### 37. Dashboard Design Best Practices
- Browser applications
- Application Performance Monitoring (APM)/Backend services
- Infrastructure (host, network, disk etc)
- Synthetic monitors (Website up?)
- Business
- Dashboard design
  - Browser applications
```bash  
|------------------------------------------------------------------------|
|             HTTT Error Rate       |        Top 10 Error Messages       |
|------------------------------------------------------------------------|
|                           Page view load time                          |
|------------------------------------------------------------------------|
|                           Page view per minute                         |
|------------------------------------------------------------------------|
| Largest Contentful Paint | First Input Delay | Cumulative Layout Shift |
|         (LCP)            |       (FID)       |         (CLS)           |
|------------------------------------------------------------------------|
```
  - APM Services
```bash  
|------------------------------------------------------------------------|
|         API Call Per Minute    |       Error Rate Per Minute           |
|------------------------------------------------------------------------|
|                               Logs                                     |
|------------------------------------------------------------------------|
|                         Hosts/Containers                               |
|------------------------------------------------------------------------|
|              CPU Usage         |           Memory Usage                | 
|------------------------------------------------------------------------|
```
  - Infrastructure
```bash  
|------------------------------------------------------------------------|
|                             Summary Info                               |
|                  (Hosts, Applications, Events, Alerts etc)             |
|------------------------------------------------------------------------|
|                             Metrics                                    |
|           (CPU, Memory, Disk Used, Disk Utilization)                   |
|------------------------------------------------------------------------|
|                         Hosts/Containers                               |
|------------------------------------------------------------------------|
|              Databases         |    Distributed Cache (and similar)    |
|------------------------------------------------------------------------|
```
  - Synthetic Monitors
```bash  
|------------------------------------------------------------------------|
|    Landing Page Up View/Alert   |     APIs Health Check Status         | 
|------------------------------------------------------------------------|
|                       Page Load Performance                            |
|------------------------------------------------------------------------|
|                      External Systems Status                           |
|------------------------------------------------------------------------|
```

### 38. The ShoeHub Global Company!
- https://github.com/aussiearef/ShoeHubV2

### 39. Connecting Grafana to Prometheus
- From web browser (localhost:3000)
  - Connections -> Add new connection -> Enter `http://localhost:9090`
- Dashboards -> New -> New Folder: enter folder name -> Create dashboard
  - Add -> Row
  - Save dashboard

### 40. Creating and Managing Dashboards in Grafana
- Add visualization in the panel

### 41. Creating Your First Panel : The Time Series Panel

### 42. Multiple and Accumulative Queries

### 43. Exercise: Display Country Data On A Graph Panel

### 44. Data Transformations

### 45. Visually Comparing Values with Pie Charts

### 46. Comparing Metric Data of Two Different Times (Time Shift)

### 47. Practice : Working with Charts and Thresholds

### 48. Thresholds in Grafana

### 49. Variables and Dynamic Dashboards

### 50. Practie Creating Dynamid Dashboards

### 51. Solved: Creating Dynamic Dashboards

### 52. Increasing the visibility of data with logarithmic scaling

### 53. Working with the Gauge and Bar Gauge Panels
- Add visualization
  - Select Gauge
  - Adjust Calculation
  - Enable/disable Show threshold labels
- To continue the practice:  
```
sudo systemctl start node.service
sudo systemctl start prometheus.service 
sudo systemctl start grafana-server.service
```

## Section 6: Working with Alerts, Notifications and Annotations in Grafana

### 54. About Alerts in Grafana
- Alerts are raised when a defined rule is violated
- Rules are defined as queries and are checked by Alert Manager
- Alerts are sent to Notification Polices
- Notification polices send notifications to contact points

### 55. Working with Alert Rules

### 56. Notification Policies and Contact Points
- Can be hooked with Jira, Teams, Email

### 57. Sending Alert Notifications to Slack

### 58. Silencing Alert Notifications

### 59. Annotations
- Describe the panels
- Hovering on chart then a window of annotation pops up:
![ch59_annotation](./ch59_annotation.png)
- ctrl + mouse will open a window of add annotation in a range:
![ch59_ranged](./ch59_ranged.png)


## Section 7: Grafana Loki

### 60. About Grafana Loki
- Open source log aggregation system designed to work seamlessly with Grafana
  - Log aggregation: collecting, storing, and querying large amounts of logs
  - Prometheus-inspired desgin: uses similar concepts and query language of Prometheus
  - Distributed architecture
  - Cost-effective storage: chunk-based storage system
- How it works?
  - Backed service writes logs at /var/logs
  - Promtail discovers the logs then pushes to Loki
  - Grafana ingests through Loki
  - Alloy may ingest from logs then push to Loki

### 61. Options of Using Grafana Loki (Cloud vs. On-Prem)
- As SaaS from grafana.com
- Promtail must be installed on every client

### 62. Installing Loki and Promtail on Linux (Ubuntu)
- Download promtail
- sudo mv promtail-linux-amd64 /usr/loca/bin/promtail
- sudo chmod +x /usr/local/bin/promtail
- cd /etc
- sudo mkdir /etc/promtail
- cd promtail
- sudo vi config.yml:
```yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: varlogs
          __path__: /var/log/*log

  - job_name: shoehub
    static_configs:
      - targets:
          - localhost
        labels:
          job: shoehub
          app: shoehub
          __path__: /var/log/shoehub/log*
```          
- cd /etc/system- sudo vi promtail.service
```ini
[Unit]
Description=Loki Promtail
After=network.target

[Service]
ExecStart=/usr/local/bin/promtail -config.file=/etc/promtail/config.yml
Restart=always

[Install]
WantedBy=default.target
```
- sudo systemctl start promtail.service
- sudo systemctl enable promtail.service

### 63. Ingesting Log Entries into Loki using Promtail
- log-gen.py
```py
import logging
from datetime import datetime
import time
import random
# Configure basic logging to write logs to a file
log_file_path = '/var/log/loki_udemy.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s level=%(levelname)s app=myapp component=%(component)s %(message)s')
def generate_log_entries():
    components = ["database", "backend"]
    for _ in range(10):
        log_level = logging.INFO if _ % 3 == 0 else logging.WARNING if _ % 3 == 1 else logging.ERROR
        component = random.choice(components)
        print(f"Generating log of type {logging.getLevelName(log_level)} with component {component}")
        if log_level == logging.INFO:
            log_message = "Information: Application running normally"
        elif log_level == logging.WARNING:
            log_message = "Warning: Resource usage high"
        else:
            log_message = "Critical error: Database connection lost"
        # Use the extra parameter to dynamically set the 'component' value
        logging.log(log_level, log_message, extra={"component": component})
        time.sleep(1)  # Sleep for 1 second between entries
if __name__ == "__main__":
    generate_log_entries()
    logging.shutdown()
```

### 64. Creating and Attaching Static Labels

### 65. Dynamic Labels: Extracting Labels from Unstructured Logs
- Add `pipeline_stage` as shown below in config.yml
```yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
- job_name: system
  static_configs:
  - targets:
      - localhost
    labels:
      job: varlogs
      __path__: /var/log/*log
      team: DevOps
      env: Prod
      component:
  pipeline_stages:
  - logfmt:
      mapping:
        component:
  - labels:
      component:
```

### 66. Visualising Loki Queries on Dashboards

### 67. Installing Grafana Loki and Promtial with Docker
- https://grafana.com/docs/loki/latest/setup/install/docker/#install-with-docker-compose
- grafana-prometheus.yml
```yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus
    networks:
      - monitoring
    ports:
      - 9090:9090

  grafana:
    image: grafana/grafana-oss  # use grafana/grafana-enterprise for Grafana Enterprise
    networks:
      - monitoring
    ports:
      - 3000:3000
    environment:
      GF_DATASOURCE: prometheus

  loki:
    image: grafana/loki:latest
    networks:
      - monitoring
    ports:
      - 3100:3100  # Loki UI port

networks:
  monitoring:
    external: false
```

## Section 8: Grafana Alloy for Logs and Opentelemetry Signals

### 68. Introduciton to Telemetry (OTel)
- Vendor-neutral and open-source observability framework
- Collects and exports metrics, logs and traces
- OTel is not an observability backend like Prometheus

### 69. The Architecture of Open-Telemetry
- Microservice: OTel SDK -> OTel exporters like Prometheus Exporter -> OTel Collectors -> Backend like Prometheus

### 70. Prometheus Remote Write for OTEL Metrics

### 71. Introduciton to Grafana Alloy
- Open source opentelemetry collector with built-in Prometheus

### 72. Intalling and Configuring Grafana Alloy on a Mac Computer

### 73. Configuring Grafana Alloy to Receive, Process and Export Opentelemetry Signals
- https://grafana.com/docs/alloy/latest/reference/components/
- config.alloy:
```json
logging {
  level  = "debug"
  format = "logfmt"
}

otelcol.receiver.otlp "default" {
  http {}
  grpc {}

  output {
    traces  = [otelcol.processor.batch.default.input]
    metrics = [otelcol.processor.batch.default_metrics.input]
    logs    = [otelcol.processor.batch.default_logs.input]
  }
}

otelcol.processor.batch "default" {
  output {
    traces = [otelcol.exporter.otlphttp.tempo.input]
  }
}

otelcol.processor.batch "default_metrics" {  
  output {
    metrics = [otelcol.exporter.prometheus.default.input]
  }
}

otelcol.processor.batch "default_logs" {
  output {
    logs = [otelcol.exporter.loki.default.input]
  }
}

otelcol.exporter.otlphttp "tempo" {
    client {
        endpoint = "http://tempo:4318"
        tls {
            insecure             = true
            insecure_skip_verify = true
        }
    }
}

otelcol.exporter.prometheus "default" {
  forward_to = [prometheus.remote_write.default.receiver]
}

prometheus.remote_write "default" {
  endpoint {
    url = "http://prometheus:9090/api/v1/write"
    basic_auth {
      username = "admin"
      password = "password"
    }
  }
}

loki.write "local" {
  endpoint {
    url = "http://loki:3100/loki/api/v1/push"
  }
}

otelcol.receiver.filelog "default" {
  include = ["/var/log/shoehub/*"]
  output {
    logs = [otelcol.exporter.loki.default.input]
  }
}

otelcol.exporter.loki "default" {
  forward_to = [loki.write.local.receiver]
}
```

### 74. Sending Metrics from a Microservice to Grafana Alloy and Prometheus

### 75. Shipping Logs to Loki with Alloy
- https://grafana.com/docs/alloy/latest/reference/components/loki/

### 76. Installing Grafana Alloy on Ubuntu
- Link broken

## Section 9: Grafana Temp: Tracing in Distributed Systems

### 77. About Tracing and Distributed Systems
- Trace: represents the overall flow or a request as it traverses through multiple services. It consists of one or more Spans. Trace has Trace ID
- Span: A single unit of work within a Trace. Information about specific action or oepration i.e., execute a database query. Span has Span ID
- Trace Context: Metadata of each trace i.e., Trace ID and Span ID
- Sampling: Reduces the volume of captured tgrace data
- For instrumentation
  - Zipkin
  - OpenTelemetry
  - NewRelic
- To capture, store, analyze & visualize trace data
  - Zipkin
  - Jaeger
  - NewRelic
  - Grafana Temp

### 78. Introduction to Grafana Tempo
- Open source high-scale distributed tracing backend
- Query Language called TraceQL
- Does not need a database
  - Local storage or object storage

### 79. Installing Grafana Tempo on MacOS

### 80. Installing Grafana Tempo on Linux
- https://github.com/grafana/tempo

### 81. Configuring Grafana Alloy to Forward Traces to Grafana Tempo
- https://grafana.com/docs/tempo/latest/?pg=oss-tempo&plcmt=quick-links
- tempo.yml
```yml
server:
  http_listen_port: 3200
  grpc_listen_port: 3300

distributor:
  receivers: 
    otlp:
      protocols:
        http:
          endpoint: "0.0.0.0:4318"  # Replace 3200 with your desired port number


compactor:
  compaction:
    block_retention: 48h                # configure total trace retention here

metrics_generator:
  registry:
    external_labels:
      source: tempo
      cluster: linux-microservices
  storage:
    path: /var/tempo/generator/wal
    remote_write:
      - url: http://admin:password@prometheus:9090/api/v1/write    # if you use as part of Docker Compose, use this line.
      # - url: http://admin:password@localhost:9090/api/v1/write   # if you run locally, use this line.
        send_exemplars: true

storage:
  trace:
    backend: local                
    local:
      path: /var/tempo/traces      # Set to correct path on your computer
    wal:
      path: /var/tempo/wal         # Set to correct path on your computer

#storage:
#  trace:
#    backend: s3                 
#    s3:
#      bucket: your-s3-bucket-name
#      region: your-region
#      access_key: your-access-key  # not needed if role_arn is used.
#      secret_key: your-secret-key  # not needed if role_arn is used.
#      role_arn: arn:aws:iam::123456789012:role/your-tempo-role

overrides:
  defaults:
    metrics_generator:
      processors: [service-graphs, span-metrics]
```    

### 82. Sending Traces from a Microservice to Grafana Tempo with OpenTelemetry

### 83. Propagating Spans in a Distributed Systems :: Service Graphs in Tempo
- Context propagation: the mechanism that passes tracing metadata - such as Trace IDs and Span IDs - between services, allowing disparate spans to be linked into a single, cohesive trace. It enables tracking a request's full flow across service boundaries,, ensuring Tempo can visualize the entire transaction's 

### 84. TraceQL for Selecting Traces and Spans in Grafana Tempo
- Inspired by PromQL and LogQL
- Selects traced based on:
  - Span and resource attributes, duration and timing
  - Basic aggregates: count(), avg(), sum(), min(), and max()
- Instinsics
  - Values that are fundamental to Spans: name, duration, status
  - All other data point names must begin with: Ex) `_http_status_code`, `_service_name`

### 85. Practice TraceQL
- https://grafana.com/docs/tempo/latest/traceql/
- https://grafana.com/docs/tempo/latest/solutions-with-traces/solve-problems-metrics-queries/
- https://view.genially.com/662de2ca7c54340013e64ade

### 86. Configuring Grafana Tempo to use AWS S3 for Storage

## Section 10: Grafana Mimir: Observability at Scale

### 87. About Grafana Mimir
- Prometheus has a single-node architecture
- Grafana Mimir extends Prometheus, not replacing it
  - Ingest data from multiple Prometheus servers or agents
- Mimir makes enterprise-gracde observability on the cloud
  - Horizontally scalable

## 88. Deploying Mimir in Monolithic Mode
- https://github.com/aussiearef/grafana-udemy/tree/main/docker
- config.yml
```yml
multitenancy_enabled: false

server:
  http_listen_port: 9009
  grpc_listen_port: 9095

common:
  storage:
    backend: filesystem
    filesystem:
      dir: /var/mimir/common

blocks_storage:
  backend: filesystem
  filesystem:
    dir: /var/mimir/blocks

ingester:
  ring:
    replication_factor: 1
```    
- `./mimir --config.file=./config.yml`

### 89. Sending and Receiving Metrics of Multiple Tenants

### 90. Configuring Mimir's Backend and Common Storages with AWS S3
- config.yml for S3:
```yml
multitenancy_enabled: false

server:
  http_listen_port: 9009
  #grpc_listen_port: 9095

common:
  storage:
    backend: s3
    s3:
      bucket_name: <your-bucket-name> # Replace with your S3 bucket name
      endpoint: s3.amazonaws.com # Use a specific endpoint if using non-default region or custom S3 provider
      region: ap-southeast-2     # Use your AWS region
      access_key_id: 
      secret_access_key: 
      insecure: false            # Set to true if using non-HTTPS S3-compatible storage (e.g., MinIO in dev)

blocks_storage:
  backend: s3
  s3:
    bucket_name: <your-blocks-bucket-name> # Replace with your S3 bucket name for blocks
    endpoint: s3.amazonaws.com
    region: ap-southeast-2
    access_key_id: 
    secret_access_key: 
    insecure: false


alertmanager_storage:
  backend: s3
  s3:
    bucket_name: <your-alertmanager-bucket-name> # Replace with your S3 bucket name for Alertmanager
    endpoint: s3.amazonaws.com
    region: ap-southeast-2
    access_key_id: 
    secret_access_key: 
    insecure: false

ingester:
  ring:
    replication_factor: 1
    kvstore:
      store: memberlist #consul, etcd, or memberlist

distributor:
  ring:
    kvstore:
      store: memberlist

store_gateway:
  sharding_ring:
    kvstore:
      store: memberlist

ruler:
  ring:
    kvstore:
      store: memberlist

querier:
  sharding_ring:
    kvstore:
      store: memberlist  # or etcd, consul
```

### 91. Deploying Mimir in Microservices Mode

### 92. Installing Minikube for Locally Deploying Mimir in Microservice Mode

### 93. Deploying Grafana Mimir to Kubernetes using Helm

### 94. Introduction to Alert Managment with Mimir
- Prometheus doesn't support multi-tenancy
- config_alerts.yml
```yml
multitenancy_enabled: true
server:
  http_listen_port: 9008
  grpc_listen_port: 9095
common:
  storage:
    backend: filesystem
    filesystem:
      dir: /var/mimir/common
blocks_storage:
  backend: filesystem
  filesystem:
    dir: /var/mimir/blocks

# --- Alerting related configurations ---
ruler:
  enable_api: false 
  alertmanager_url: http://localhost:9008/alertmanager  # do not set if you deploy with Helm
  ring:
    kvstore:
      store: memberlist
  
ruler_storage:
  backend: filesystem
  filesystem:
    dir: /var/mimir/ruler
  storage_prefix: ""
  local:
      directory: ""

alertmanager:
  enable_api: true
  sharding_ring:
    kvstore:
      store: memberlist
      prefix: alertmanagers/
  fallback_config_file: /configs/fallback_alertmanager.yaml

alertmanager_storage:
  filesystem:
    dir: alertmanager
  storage_prefix: ""
  

ingester:
  ring:
    replication_factor: 1
```    
- Steps
  1. Create one rules.yml per tenant
  2. Use mimirtool to load the rules.yml to Ruler
  3. Create one alert manager.yml per tenant, or a fallback alertmanager.yml

### 95. Configuring Ruler, Alert Manager and Mimir for Alerting
- tenant-rules.yml:
```yml
groups:
  - name: shoehub.rules
    interval: 1m
    rules:
      - alert: HighBootsSalesRate
        expr: sum(rate(shoehub_sales{ShoeType="Boots"}[1m])) > 1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High sales rate for Boots detected"
          description: "The sales rate for Boots ({{ $value }} per second) exceeds 10."
```

### 96. Mimirtool for Configuring Alert Manager
- https://github.com/grafana/mimir/releases

## Section 11: Smart Observability: Integrating AI with Grafana

### 97. Applicaiton of AI in Grafana Cloud and Grafana OSS
- Challenges in traditional observability
  - High volume of metrics, logs, and traces
  - Alert fatigue from too many noisy alerts
  - Difficult root cause analysis
  - Static thresholds miss dynamic patterns
- How AI helps
  - Detects anomalies in real-time
  - Correlates metrics, logs, and traces
  - Predicts incidents before they occur
  - Summarizes logs and generates insights
- Grafana Cloud AI capabilities
  - Adaptive Alerts with ML models
  - Automatic anomaly detection
  - ML powered alert tuning
  - Event correlation engine
- Grafana OSS limitations
  - No built-in ML models or adaptive alerts
  - No advanced log summarisation
- Application of AI in OSS Grafana
  - Free AI tools from ChatGPT, Google Gemini, ...
  - Develop a Grafana plugin to use API    

### 98. Effective Prompt Engineering to Bolster Observability with AI
- Use effective prompt engineering

### 99. Using Plugins to Leverage the Power AI in Grafana Effectively
- https://github.com/grafana/plugin-tools
- https://github.com/grafana/grafana-llm-app

## Section 12: Integration With Other DataSources

### 100. Integration of Grafana with MySQL
- Query must return 3 mandatory columns
  - Metric Name: Metric
  - Metric Value: value
  - Unix Time Stamp: time_sec

### 101. Integration of Grafana with SQL Server

### 102. Integration of Grafana with AWS Cloudwatch

### 103. Monitoring Google Cloud Platform with out-of-the-box dashboards

## Section 13: Administration of Grafana

### 104. Overview of Administration in Grafana
- Grafana
  - Organization
    - User
    - Team 
      - User
      - Service Account
    - Dashboard

### 105. Working with Organisations, Teams and Users in Grafana
- Administratoin
  - Users and access
    - Create a new user

### 106. Authenticating Users with Google

### 107. Authenticating Users with Active Directory

### 108. Installing Plugins in Grafana

## Section 14: Highly Available and Scalable Grafana

### 109. Deploying Grafana for High Availability (HA)
- https://arefkarimi.com/install-grafana-with-high-availability-manually-with-mysql-in-a-multi-server-on-premise-environment/
- Multiple instances of Grafana
  - They must be identical, having same ini files
- Web browser may access one of them through load balancer

### 110. Deploying Grafana for Scalability

## Section 15: The Killer Coda Envrionment to Reinforce What You've Learned
- https://killercoda.com/aref-karimi/course/grafana

### 111. Grafana and Prometheus on Killer Coda

## Section 16: Bonus Items

### 112. Infographics

### 113. Bonus Lecture: New Relic One Observability Platform


