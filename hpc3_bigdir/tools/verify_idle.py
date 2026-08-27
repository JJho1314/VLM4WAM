import os, subprocess, numpy as np, imageio_ffmpeg
D=os.environ["DSDIR"]; FF=imageio_ffmpeg.get_ffmpeg_exe(); DW,DH=432,240
def prof(stem):
    cmd=[FF,"-v","error","-i",f"{D}/videos/{stem}.mp4","-vf",f"scale={DW}:{DH}","-pix_fmt","gray","-f","rawvideo","-"]
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=10**8)
    prev=None;sc=[];fsz=DW*DH
    while True:
        b=p.stdout.read(fsz)
        if len(b)<fsz:break
        g=np.frombuffer(b,np.uint8).reshape(DH,DW).astype(np.float32)
        if prev is not None: sc.append(float((np.abs(g-prev)>12).mean())*100)
        prev=g
    p.stdout.close();p.wait()
    return np.array(sc)
for stem in ["episode_027256_left_external","episode_029216_right_external","episode_009701_left_external"]:
    f=prof(stem); T=len(f)+1
    # show max motion in first 80% vs last 20%
    cut=int(0.8*len(f))
    print(f"{stem} T={T}: max_motion first80%={f[:cut].max():.2f}%  last20%={f[cut:].max():.2f}%  mean_all={f.mean():.2f}%")
    # print last 12 steps
    print("   last12 motion%:", " ".join(f"{x:.1f}" for x in f[-12:]))
