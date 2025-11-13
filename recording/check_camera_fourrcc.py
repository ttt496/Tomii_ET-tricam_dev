import cv2
import sys

# OpenCVの警告を抑制
cv2.setLogLevel(0)

cams = range(0, 4)
# バックエンドのリスト: (名前, OpenCVのバックエンドID)
# DSHOW = DirectShow (Windows), MSMF = Media Foundation (Windows)
backends = [('DSHOW', getattr(cv2, 'CAP_DSHOW', 700)), ('MSMF', getattr(cv2, 'CAP_MSMF', 1400))]
fmts = ['MJPG', 'YUY2']
res = [(1980, 1080), (1280, 720), (640, 480)]
fps_list = [30, 60]

print('probe start')

for i in cams:
    for backend_name, backend_id in backends:
        for f in fmts:
            for w, h in res:
                for r in fps_list:
                    cap = cv2.VideoCapture(i, backend_id)
                    if not cap.isOpened():
                        cap.release()
                        print(f'cam{i} {backend_name} req={f} {w}x{h}@{r} -> NG')
                        continue
                    
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*f))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    cap.set(cv2.CAP_PROP_FPS, r)
                    
                    ok, frm = cap.read()
                    
                    if ok:
                        # 実際の設定値を取得
                        v = int(cap.get(cv2.CAP_PROP_FOURCC))
                        fourcc_int = v
                        fourcc_str = ''.join([chr((v >> (8 * k)) & 255) for k in range(4)])
                        # 制御文字や非表示文字を置換
                        fourcc_display = ''.join([c if 32 <= ord(c) < 127 else '?' for c in fourcc_str])
                        
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        print(f'cam{i} {backend_name} req={f} {w}x{h}@{r} -> OK '
                              f'actual=FourCC:{fourcc_display}({fourcc_int:08x}) '
                              f'{actual_w}x{actual_h}@{actual_fps:.1f}fps')
                    else:
                        print(f'cam{i} {backend_name} req={f} {w}x{h}@{r} -> NG')
                    
                    cap.release()

print('done')