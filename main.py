import psutil

def get_cpu_usage():
    cpu_usage_persent = psutil.cpu_percent(interval=1, percpu=True)
    cpu_count = psutil.cpu_count()
    cpu_stats = psutil.cpu_stats()
    return cpu_usage_persent , cpu_stats, cpu_count




print(get_cpu_usage())