BillSums dataset
https://huggingface.co/datasets/FiscalNote/billsum/

### Run and build if any changes
```bash
docker compose up -–build
```

### Just run
```bash
docker compose up
```
```bash
docker compose up -d
```

### Destroy containers
```bash
docker compose down
```

### Just stop
```bash
docker compose stop
```

### Resume
```bash
docker compose start
```

### Check if running
```bash
docker compose ps
```

### To see the logs only from the main node
```bash
docker compose logs -f orchestrator_logic
```

### Stats of containers
```bash
docker stats
```

Using hugging face auth token so you might need a new one in case of issues


Run docker compose for the execution, in parallel in another window run the docker memory tracker
```bash
python monitor.py
```