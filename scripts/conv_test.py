import csv

csv_path = "/srv/home/rangrej/CS759Project/results/bench_121243/baseline.csv"

with open(csv_path, newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        if (
            row["impl"] in ("cuda", "cuda_opt")
            and row["H"] == "1024"
            and row["K"] == "3"
        ):
            print(
                f'{row["impl"]}: '
                f'total={row["median_ms"]} ms, '
                f'H2D={row["h2d_ms"]} ms, '
                f'kernel={row["kernel_ms"]} ms, '
                f'D2H={row["d2h_ms"]} ms, '
                f'pass={row["pass"]}'
            )
