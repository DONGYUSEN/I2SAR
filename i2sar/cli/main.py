from __future__ import annotations

import argparse
import sys
import time
import os
import tempfile

from i2sar.project import Project
from i2sar.rtc.strip_rtc import run_strip_rtc, get_dem_for_extent
from i2sar.workflow import insar_workflow_specs, rtc_workflow_specs, run_empty_task, scene_workflow_specs
from i2sar.io import import_scene


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="i2sar")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-project")
    create.add_argument("root")
    create.add_argument("--name", required=True)

    run_empty = subparsers.add_parser("run-empty")
    run_empty.add_argument("project")
    run_empty.add_argument("--workflow", choices=["scene", "insar", "rtc"], required=True)
    run_empty.add_argument("--owner-id", required=True)
    run_empty.add_argument("--no-resume", action="store_true")

    strip_rtc = subparsers.add_parser("strip-rtc")
    strip_rtc.add_argument("source", help="Source data path (ZIP/TIFF/directory)")
    strip_rtc.add_argument("--sensor", default="auto", help="Sensor type: tianyi, lutan, sentinel1, auto")
    strip_rtc.add_argument("--output-dir", required=True, help="Output directory")
    strip_rtc.add_argument("--project-dir", help="Project directory")
    strip_rtc.add_argument("--dem-cache", default="~/Temp/dem", help="DEM cache directory")
    strip_rtc.add_argument("--output-resolution", type=float, help="Output resolution in meters")
    strip_rtc.add_argument("--full-resolution", action="store_true", help="Process at full resolution")
    strip_rtc.add_argument("--download-dem", action="store_true", default=True, help="Download missing DEM")
    strip_rtc.add_argument("--nalks", type=int, default=1, help="方位向视数 (azimuth looks)")
    strip_rtc.add_argument("--nrlks", type=int, default=1, help="距离向视数 (range looks)")

    download_dem = subparsers.add_parser("download-dem")
    download_dem.add_argument("--source", help="Source data to extract bounds from")
    download_dem.add_argument("--lat", type=float, help="Center latitude")
    download_dem.add_argument("--lon", type=float, help="Center longitude")
    download_dem.add_argument("--radius", type=float, default=0.1, help="Search radius in degrees")
    download_dem.add_argument("--bounds", nargs=4, type=float, metavar=("S", "N", "W", "E"),
                      help="Bounding box: south north west east")
    download_dem.add_argument("--output", required=True, help="Output DEM HDF5 file")
    download_dem.add_argument("--sensor", default="auto", help="Sensor type: tianyi, lutan, sentinel1, auto")
    download_dem.add_argument("--dem-cache", default="~/Temp/dem", help="DEM cache directory")

    multilook_parser = subparsers.add_parser("multilook")
    multilook_parser.add_argument("input", help="输入TIFF文件")
    multilook_parser.add_argument("output", help="输出TIFF文件")
    multilook_parser.add_argument("--nalks", type=int, default=1, help="方位向视数")
    multilook_parser.add_argument("--nrlks", type=int, default=1, help="距离向视数")
    multilook_parser.add_argument("--chunk-lines", type=int, default=1000, help="分块行数")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "create-project":
        Project.create(args.root, args.name)
        return 0

    if args.command == "run-empty":
        project = Project.open(args.project)
        if args.workflow == "scene":
            specs = scene_workflow_specs(args.owner_id)
        elif args.workflow == "insar":
            specs = insar_workflow_specs(args.owner_id)
        else:
            specs = rtc_workflow_specs(args.owner_id)
        for spec in specs:
            run_empty_task(project, spec, resume=not args.no_resume)
        return 0

    if args.command == "strip-rtc":
        from pathlib import Path
        
        output_dir = Path(args.output_dir)
        dem_cache = Path(os.path.expanduser(args.dem_cache))
        
        if args.project_dir:
            project_dir = Path(args.project_dir)
        else:
            project_dir = output_dir / "project"
        
        print(f"Processing {args.source}")
        print(f"Output directory: {output_dir}")
        print(f"DEM cache: {dem_cache}")
        print(f"Full resolution: {args.full_resolution}", file=sys.stderr)
        sys.stderr.flush()
        
        t0 = time.time()
        
        try:
            print("Starting run_strip_rtc...", file=sys.stderr)
            sys.stderr.flush()
            
            result = run_strip_rtc(
                source=args.source,
                sensor=args.sensor,
                output_dir=output_dir,
                project_dir=project_dir,
                dem_cache=dem_cache,
                output_resolution=args.output_resolution,
                full_resolution=args.full_resolution,
                download_dem=args.download_dem,
                nalks=args.nalks,
                nrlks=args.nrlks,
            )
            elapsed = time.time() - t0
            print(f"\nCompleted in {elapsed/60:.1f} minutes")
            print(f"RTC PNG: {result.rtc_png}")
            print(f"KML: {result.kml_file}")
            print(f"EPSG: {result.epsg}")
            return 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            return 1

    if args.command == "download-dem":
        from pathlib import Path
        import h5py
        
        output_path = Path(args.output)
        dem_cache = Path(os.path.expanduser(args.dem_cache))
        
        south, north, west, east = None, None, None, None
        
        if args.source:
            print(f"Extracting bounds from: {args.source}")
            with tempfile.TemporaryDirectory() as tmpdir:
                project_path = Path(tmpdir) / "project.h5"
                project = Project.create(project_path, name="dem_download")
                import_result = import_scene(project, args.source, sensor=args.sensor)
                
                scene_h5_path = import_result.scene_path
                with h5py.File(scene_h5_path, "r") as h5:
                    if "grid" in h5:
                        grid_data = h5["grid"]
                        lats = grid_data["latitude"][:]
                        lons = grid_data["longitude"][:]
                    elif "slc" in h5 and "grid" in h5["slc"]:
                        grid_data = h5["slc"]["grid"]
                        lats = grid_data["latitude"][:]
                        lons = grid_data["longitude"][:]
                    else:
                        raise ValueError(f"No grid found in {scene_h5_path}")
                    
                    south = float(lats.min())
                    north = float(lats.max())
                    west = float(lons.min())
                    east = float(lons.max())
                
                margin = 0.05
                south -= margin
                north += margin
                west -= margin
                east += margin
                print(f"Extracted bounds: S={south:.4f} N={north:.4f} W={west:.4f} E={east:.4f}")
        elif args.bounds:
            south, north, west, east = args.bounds
            print(f"Using bounds: S={south:.4f} N={north:.4f} W={west:.4f} E={east:.4f}")
        elif args.lat and args.lon:
            south = args.lat - args.radius
            north = args.lat + args.radius
            west = args.lon - args.radius
            east = args.lon + args.radius
            print(f"Using center: ({args.lat}, {args.lon}), radius: {args.radius}")
            print(f"Bounding box: S={south:.4f} N={north:.4f} W={west:.4f} E={east:.4f}")
        else:
            print("Error: Please specify --source, --bounds, or --lat/--lon", file=sys.stderr)
            return 1
        
        print(f"Downloading DEM to: {output_path}")
        
        try:
            result = get_dem_for_extent(
                south=south,
                north=north,
                west=west,
                east=east,
                output_path=output_path,
                dem_cache=dem_cache,
            )
            print(f"DEM downloaded successfully!")
            print(f"  File: {output_path}")
            return 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            return 1

    if args.command == "multilook":
        from i2sar.processing.multilook import multilook_chunked

        print(f"输入: {args.input}")
        print(f"输出: {args.output}")
        print(f"方位向视数: {args.nalks}, 距离向视数: {args.nrlks}")

        t0 = time.time()
        try:
            success = multilook_chunked(
                input_path=args.input,
                output_path=args.output,
                nalks=args.nalks,
                nrlks=args.nrlks,
                chunk_lines=args.chunk_lines
            )
            elapsed = time.time() - t0
            print(f"\nCompleted in {elapsed:.1f} seconds")
            return 0 if success else 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            return 1

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())