import argparse
import pickle
import subprocess
from pathlib import Path
from typing import List, Tuple

def download(video_id:str, video_dir:Path):
    cmd = f'yt-dlp --id -P {str(video_dir)} -f mp4 -- {video_id}'
    # %(id)s.%(ext)s
    print(f"{cmd=}")
    subprocess.call(cmd.split())

def extract_times(daly, video_id:str, minimum_second:int)->List[Tuple[str, float, float]]:
    dic = daly["annot"][f"{video_id}.mp4"]["annot"]
    out = []
    assert type(dic) == dict
    for annot in dic.keys():
        assert type(dic[annot]) == list
        for item in dic[annot]:
            # item["flags"]
            start = item["beginTime"]
            end = item["endTime"]
            if type(start) == int: start = float(start)
            if type(end) == int: end = float(end)
            assert type(start) == float, f"{video_id=} {type(start)=} {start=}"
            assert type(end) == float, f"{video_id=} {type(end)=} {end=}"
            if end - start < minimum_second: continue
            out.append((annot, start, end))
    return out

def split_video(video_path:Path, start:float, end:float, new_video_path:Path):
    assert video_path.exists()
    duration = end - start
    assert duration > 0
    cmd = f"ffmpeg -ss {start} -i {str(video_path)} -t {duration} -c copy {str(new_video_path)}"
    print(f"{cmd=}")
    subprocess.call(cmd.split())


def main(daly_pkl_path:str, video_dir:str, splited_video_dir:str, minimum_second:int):
    dp = Path(daly_pkl_path)
    assert dp.exists()

    vd = Path(video_dir)
    if not vd.exists():
        vd.mkdir(parents=True)

    svd = Path(splited_video_dir)
    if not svd.exists():
        svd.mkdir(parents=True)

    with open(dp, "rb") as f:
        daly = pickle.load(f, encoding="latin1")

    video_count = 0
    section_count = 0
    for v in daly["annot"].keys():
        video_id = v.replace(".mp4", "")
        times = extract_times(daly, video_id, minimum_second)
        if len(times) > 0:
            print(f"############ {video_id=} ############")
            print(*times, sep="\n")
            download(video_id, vd)
            video_path = Path(vd, f"{video_id}.mp4")
            if video_path.exists():
                for i, (annot, start, end) in enumerate(times):
                    new_video_name = f"{video_id}_{int(start)}_{i}_{annot}.mp4"
                    new_video_path = Path(svd, new_video_name)
                    split_video(video_path, start, end, new_video_path)
                video_count += 1
                section_count += len(times)
    print(f"{video_count=} {section_count=}")


if __name__ == "__main__":
    dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--daly_pkl_path', required=True, default=f"{Path(dir, 'daly1.1.0.pkl')}")
    parser.add_argument('--video_dir', required=True, default=f"{Path(dir, '_0_raw')}")
    parser.add_argument('--splited_video_dir', required=True, default=f"{Path(dir, '_1_split')}")
    parser.add_argument('--minimum_second', type=int, default=10)
    args = parser.parse_args()
    main(**vars(args))