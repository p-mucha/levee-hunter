from pathlib import Path
import requests
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from time import sleep
import pandas as pd
import json
import argparse

# https://apps.nationalmap.gov/tnmaccess/#/product

# example of a single request
# https://tnmaccess.nationalmap.gov/api/v1/products?&datasets=National%20Elevation%20Dataset%20(NED)%201%20arc-second%20Current&prodFormats=GeoTIFF&bbox=-121.58826,37.96628,-121.51895,38.01853

DATASETS = ','.join([
    #"National Elevation Dataset (NED) 1 arc-second Current",
    "National Elevation Dataset (NED) 1/3 arc-second Current",
    #"National Elevation Dataset (NED) 1/9 arc-second",
    #"Original Product Resolution (OPR) Digital Elevation Model (DEM)",
    #"Digital Elevation Model (DEM) 1 meter", # need --cellSize 0.3
    ])
PRODFORMAT="GeoTIFF"

LEVEE_PATH = Path("/share/gpu1/lshi/fathom/data/levee/downloads/levees.gpkg")

class ProgressParallel(Parallel):
    def __init__(
        self, use_tqdm=True, total=None, desc=None, leave=True, *args, **kwargs
    ):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        self._leave = leave
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm,
            total=self._total,
            desc=self._desc,
            leave=self._leave,
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def query_geodata(datasets, prodFormats, bbox=""):
    request_str = f"https://tnmaccess.nationalmap.gov/api/v1/products?&datasets={datasets}&prodFormats={prodFormats}"
    if bbox:
        request_str += f"&bbox={bbox}"
        print("bbox:", bbox)
    response = requests.get(request_str)
    try:
        response.raise_for_status()
        res = response.json()
        print(res['messages'][0])
        if res["total"]>len(res["items"]):
            raise Warning(f"Query returns {res['total']} items, exceeding the limit of {len(res['items'])}. Consider to descrease the cell size.")
        return res
    except Exception as e:
        print(e)
        print("Something went wrong. | Response:", response.text)
        return None

def download_geotiff(url, download_path):
    local_filename = url.split('/')[-1]
    r = requests.get(url)
    with open(download_path / f"{local_filename}", "wb") as outfile:
        outfile.write(r.content)

def chunker(lst, chunksize):
    for i in range(0, len(lst), chunksize):
        yield lst[i : i + chunksize]

def make_task_list(cell_size, levee_path, bbox = []):
    import geopandas as gpd
    from shapely.geometry import box
    import numpy as np

    def divide(total_bounds, length):
        xs = np.arange(total_bounds[0] + length/2, total_bounds[2], step=length)
        ys = np.arange(total_bounds[1] + length/2, total_bounds[3], step=length)
        if (len(xs) == 0) or (len(ys) == 0):
            print("The area is too small to divide, using the whole area")
            xmin, ymin, xmax, ymax = total_bounds
            return gpd.GeoSeries(box(xmin, ymin, xmax, ymax))
        else:
            print("Dividing the area into", len(xs), "x", len(ys), "squares")
            xv, yv = np.meshgrid(xs, ys)
            combinations = np.array(list(zip(xv.flatten(), yv.flatten())))
            squares = gpd.points_from_xy(combinations[:, 0], combinations[:, 1]).buffer(length/2, cap_style=3)
            return gpd.GeoSeries(squares)
    
    levee_gdf = gpd.read_file(levee_path, layer="System")
    if bbox is not None and len(bbox) == 4:
        total_bounds = np.array(bbox)
        print("Using provided total bounds")
    else:
        total_bounds = levee_gdf[["name", "geometry"]].geometry.total_bounds
        print("No valid total bounds provided. Using the bounds of levee data.")

    print("Total bounds:", total_bounds)
    grid = divide(total_bounds, cell_size)
    mask = grid.apply(lambda b: levee_gdf['geometry'].intersects(b).any())
    return list(grid[mask].bounds.astype(str).stack().groupby(level=0).agg(",".join))

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--taskList", type=str, help="Path to a txt file containing one bounding box per line")
    parser.add_argument("-g", "--generateTaskListOnly", action="store_true", default=False, help="Generate task list only")
    parser.add_argument("-o", "--outputPath", type=str, default="Downloads", help="Path to the folder for downloaded files")
    parser.add_argument("-b", "--totalBounds", type=float, nargs=4, help="Total bounds of the area to download: xmin ymin xmax ymax. E.g. -121.58826 37.96628 -120.51895 39.01853")
    parser.add_argument("--dryRun", action="store_true", default=False, help="Dry run, send queries but no download")
    parser.add_argument("--cellSize", type=float, default=0.5, help="Cell size for the grid when dividing downloading task")
    parser.add_argument("--chunkSize", type=int, default=10, help="Number of queries to send in each downloading batch")
    parser.add_argument("--nJobs", type=int, default=2, help="Number of parallel download jobs")
    args = parser.parse_args()

    download_path = Path(args.outputPath)
    download_path.mkdir(exist_ok=True, parents=True)
    if args.taskList is not None:
        # take the downloading request from a txt file
        with open(args.taskList, "r") as file:
            task_list = [line.strip() for line in file if not line.lstrip().startswith("#")]
    else:
        # build the full downloading request according to the coverage of levee data
        task_list = make_task_list(cell_size = args.cellSize, levee_path = LEVEE_PATH, bbox = args.totalBounds)
        with open(download_path / 'task_list.txt', 'w') as f:
            for task in task_list:
                f.write(f"{task}\n")
        print(f"Task list generated. {len(task_list)} tasks saved to {download_path}/task_list.txt")

    if args.generateTaskListOnly:
        if args.taskList is not None:
            print(f"Task list loaded from {args.taskList}. No need to generate.")
        return

    source_id_set = set() # to avoid duplicate download

    for chunk in chunker(task_list, args.chunkSize):
        jobs = []
        total_size = 0;
        for task in chunk:
            res = query_geodata(DATASETS, PRODFORMAT , task)
            if res is not None and res["total"] > 0:
                items_df = pd.json_normalize(res["items"])
                duplicate_id_mask = items_df['sourceId'].isin(source_id_set)
                source_id_set.update(items_df['sourceId'].tolist())
                filtered_urls = items_df.loc[~duplicate_id_mask, 'downloadURL'].tolist()
                total_size += items_df.loc[~duplicate_id_mask, 'sizeInBytes'].sum()*1e-6
                jobs += [delayed(download_geotiff)(url, download_path) for url in filtered_urls]
        print(f"Adding {len(jobs)} items to download job list. Total size {total_size:.2f} MB")

        if not args.dryRun:
            with parallel_backend(backend="threading", n_jobs=args.nJobs):
                ProgressParallel(use_tqdm=True, total=len(jobs), desc=f"Downloading {PRODFORMAT}")(jobs)
        else:
            print("Dry run, no download")
            sleep(10)

if __name__ == "__main__":
    run()