"""Isolated visual acceptance data; never activates production materials."""
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'.demo/image-acceptance'


def prepare():
    (DEST/'configs').mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'configs/default.yaml',DEST/'configs/default.yaml')
    for folder in ('reports/2026-11-14','data/raw/2026-11-14','data/cache/diagnostics'):
        shutil.copytree(ROOT/folder,DEST/folder,dirs_exist_ok=True)
    for name in ('data/forecasts/2026-11-14.json','data/history/macro.parquet','data/history/earnings.parquet'):
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    samples=DEST/'samples';samples.mkdir(exist_ok=True)
    title=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',40)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',27)
    for ticker,values in [('AAPL',['Revenue fell 18% year over year.','Operating profit fell 25%.','Management reported lower demand.']),
                          ('MSFT',['Revenue fell 8% year over year.','Operating profit fell 12%.','Operating costs increased.'])]:
        im=Image.new('RGB',(1100,650),'white');d=ImageDraw.Draw(im)
        d.text((55,45),ticker+' | Financial update',font=title,fill='#162a44')
        d.text((55,120),'Illustrative test document | 14 November 2026',font=font,fill='#666666')
        for i,value in enumerate(values):
            d.rectangle((45,205+i*95,1055,285+i*95),fill='#f0f3f7')
            d.text((65,225+i*95),value,font=font,fill='#222222')
        d.text((55,560),'Synthetic figures for software testing only.',font=font,fill='#666666')
        im.save(samples/(ticker+'-document.png'))
    print(DEST)


if __name__=='__main__': prepare()
