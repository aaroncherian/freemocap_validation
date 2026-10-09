#!/usr/bin/env python3
r"""Read paired DLC CSVs from a folder or ZIP and plot digital marker removal.

uv run --with pandas --with numpy --with matplotlib python plot_marker_removal.py "C:\path\outputs.zip" --output marker_analysis

Uses raw CSVs, ignoring *_filtered.csv. Keeps all frames without smoothing,
alignment, rescaling, or confidence filtering. Edit VIEW_CAMERAS and EXCLUDED
below to change the camera mapping or annotation coverage. No other scripts needed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickletools
import re
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# Displayed view number -> camera number in the input filenames.
VIEW_CAMERAS = {1: 1, 2: 4, 3: 5, 4: 6}
# Camera-specific exclusions because these landmarks were not labeled in that view.
EXCLUDED = {4: {'right_heel'}, 5: {'right_foot_index'}}
LANDMARKS = {'right_knee': 'Knee', 'right_heel': 'Heel', 'right_foot_index': 'Toe', 'right_ankle': 'Ankle'}
COLORS = {'original': '#17659a', 'edited': '#ce552d'}
FILENAME = re.compile(r'^cam_?(\d+)_(no_markers|markers)DLC.*', re.IGNORECASE)


def literal_metadata(blob):
    """Read a few literal metadata fields; never instantiate pickled objects."""
    ops = list(pickletools.genops(blob))
    fields = {'fps': 1, 'nframes': 1, 'batch_size': 1, 'frame_dimensions': 2}
    result = {}
    for i, (op, arg, _) in enumerate(ops):
        if op.name not in {'SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'BINUNICODE8'}:
            continue
        following = [(p.name, a) for p, a, _ in ops[i+1:i+6] if p.name != 'MEMOIZE']
        if arg in fields:
            values = [a for name, a in following[:fields[arg]] if name in {'BININT', 'BININT1', 'BININT2', 'INT', 'BINFLOAT', 'FLOAT'}]
            if len(values) == fields[arg]:
                result[arg] = values[0] if len(values) == 1 else values
        elif arg == 'cropping' and following and following[0][0] in {'NEWTRUE', 'NEWFALSE'}:
            result[arg] = following[0][0] == 'NEWTRUE'
    return result


def input_files(source):
    """Yield relevant bytes without extracting an archive or loading videos/HDF5."""
    def relevant(name):
        return FILENAME.match(Path(name).name) and (name.lower().endswith('.csv') or name.lower().endswith('_meta.pickle'))
    if source.is_dir():
        for path in sorted(source.rglob('*')):
            if path.is_file() and relevant(path.name):
                yield path.relative_to(source).as_posix(), path.read_bytes()
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                if relevant(name):
                    yield name, archive.read(name)
    else:
        raise ValueError('Input must be an existing folder or ZIP containing the DLC CSV files.')


def load_inputs(source):
    tables, info, metadata = {}, {}, {}
    for name, blob in input_files(source):
        match = FILENAME.match(Path(name).name)
        key = (int(match[1]), match[2].lower())
        if key[0] not in VIEW_CAMERAS.values():
            continue
        if name.lower().endswith('_filtered.csv'):
            continue
        if name.lower().endswith('_meta.pickle'):
            if key in metadata:
                raise ValueError(f'Duplicate metadata for camera/condition {key}; use one recording per input folder.')
            metadata[key] = literal_metadata(blob)
            continue
        if key in tables:
            raise ValueError(f'Duplicate raw CSV for camera/condition {key}; use one recording per input folder.')
        table = pd.read_csv(io.BytesIO(blob), header=[0,1,2], index_col=0)
        if not len(table) or table.index.has_duplicates or table.columns.has_duplicates:
            raise ValueError(f'Empty data or duplicate frames/columns: {name}')
        if table.columns.nlevels != 3 or table.columns.get_level_values(0).nunique() != 1:
            raise ValueError(f'Expected DLC scorer/bodypart/coordinate headers and one scorer: {name}')
        indices = pd.to_numeric(table.index, errors='raise').to_numpy(dtype=float)
        if not np.isfinite(indices).all() or not np.equal(indices,np.floor(indices)).all() or not np.all(np.diff(indices)>0):
            raise ValueError(f'Frame indices must be finite increasing integers: {name}')
        table.index = indices.astype(int)
        tables[key] = table.apply(pd.to_numeric, errors='raise')
        info[key] = {'file': name, 'sha256': hashlib.sha256(blob).hexdigest(), 'frames': len(table)}
    missing = [(camera, condition) for camera in VIEW_CAMERAS.values() for condition in ['markers','no_markers'] if (camera,condition) not in tables]
    if missing:
        raise ValueError(f'Missing raw CSV pairs: {missing}. Filtered files are not substituted.')
    return tables, info, metadata


def compare_views(tables, cutoff):
    summaries, frames = [], []
    for view, camera in VIEW_CAMERAS.items():
        original, edited = tables[(camera,'markers')], tables[(camera,'no_markers')]
        if not original.index.equals(edited.index) or not original.columns.equals(edited.columns):
            raise ValueError(f'Camera {camera}: original/edited indices or model/landmark columns differ.')
        scorer = original.columns.get_level_values(0)[0]
        for landmark, label in LANDMARKS.items():
            if landmark in EXCLUDED.get(camera,set()):
                continue
            if landmark not in original.columns.get_level_values(1):
                raise ValueError(f'Camera {camera}: missing required landmark {landmark}.')
            a, b = original[scorer][landmark], edited[scorer][landmark]
            if not {'x','y','likelihood'}.issubset(a.columns):
                raise ValueError(f'Camera {camera}, {landmark}: missing coordinates or likelihood.')
            values = np.column_stack([a[['x','y','likelihood']], b[['x','y','likelihood']]])
            if not np.isfinite(values).all():
                raise ValueError(f'Camera {camera}, {landmark}: nonfinite predictions; inspect these rather than silently dropping frames.')
            if not a.likelihood.between(0,1).all() or not b.likelihood.between(0,1).all():
                raise ValueError(f'Camera {camera}, {landmark}: likelihood outside 0-1.')
            dx, dy = b.x-a.x, b.y-a.y
            distance = np.hypot(dx,dy)
            frames.append(pd.DataFrame({'view': view, 'camera': camera, 'landmark': landmark, 'frame_index': original.index,
                'original_x': a.x.to_numpy(), 'original_y': a.y.to_numpy(), 'edited_x': b.x.to_numpy(), 'edited_y': b.y.to_numpy(),
                'dx_px': dx.to_numpy(), 'dy_px': dy.to_numpy(), 'displacement_px': distance.to_numpy(),
                'original_likelihood': a.likelihood.to_numpy(), 'edited_likelihood': b.likelihood.to_numpy(),
                'likelihood_change': (b.likelihood-a.likelihood).to_numpy()}))
            summaries.append({'view': view, 'camera': camera, 'landmark': landmark, 'frames': len(a),
                'mean_displacement_px': distance.mean(), 'median_displacement_px': distance.median(),
                'p95_displacement_px': distance.quantile(.95), 'max_displacement_px': distance.max(),
                'max_displacement_frame_index': int(distance.idxmax()), 'mean_dx_px': dx.mean(), 'mean_dy_px': dy.mean(),
                'original_mean_likelihood': a.likelihood.mean(), 'edited_mean_likelihood': b.likelihood.mean(),
                'mean_likelihood_change': (b.likelihood-a.likelihood).mean(), 'likelihood_cutoff': cutoff,
                'original_frames_below_cutoff': int((a.likelihood<cutoff).sum()), 'edited_frames_below_cutoff': int((b.likelihood<cutoff).sum())})
    return pd.DataFrame(summaries), pd.concat(frames, ignore_index=True)


def format_axis(ax):
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))


def legend(fig, cutoff=None):
    handles = [Line2D([],[],color=COLORS['original'],lw=2,label='Original'),
               Line2D([],[],color=COLORS['edited'],lw=2,ls='--',label='Digitally edited')]
    if cutoff is not None:
        handles.append(Line2D([],[],color='#888888',ls=':',label=f'Likelihood {cutoff:g}'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,.995),ncol=len(handles),frameon=False)


def save_figure(fig, output, stem, dpi):
    fig.savefig(output/f'{stem}.png',dpi=dpi)
    fig.savefig(output/f'{stem}.pdf')


def plot_xy(data, output, dpi):
    with PdfPages(output/'all_views_xy_overlays.pdf') as pdf:
        for view in VIEW_CAMERAS:
            selected = data[data.view==view]
            landmarks = [key for key in LANDMARKS if key in selected.landmark.unique()]
            fig, axes = plt.subplots(len(landmarks),2,figsize=(11,2.2*len(landmarks)+.65),sharex=True,squeeze=False)
            for row, landmark in enumerate(landmarks):
                part = selected[selected.landmark==landmark]
                for col, coord in enumerate(['x','y']):
                    ax = axes[row,col]
                    ax.plot(part.frame_index,part[f'original_{coord}'],color=COLORS['original'],lw=1.8)
                    ax.plot(part.frame_index,part[f'edited_{coord}'],color=COLORS['edited'],ls='--',lw=1.5)
                    ax.set_title(f'{LANDMARKS[landmark]} — {coord.upper()}',loc='left',fontsize=11,weight='bold')
                    ax.set_ylabel(f'{coord.upper()} position (px)')
                    format_axis(ax)
            for ax in axes[-1]:
                ax.set_xlabel('Frame #')
            # Small view identifier only: no descriptive overall title or footer.
            fig.text(.075,.98,f'View {view}',ha='left',va='top',fontsize=12,weight='bold')
            legend(fig)
            fig.tight_layout(rect=(0,0,1,.94),h_pad=1.2)
            save_figure(fig,output,f'view_{view}_xy_overlay',dpi)
            pdf.savefig(fig)
            plt.close(fig)


def plot_overview(data, summary, output, kind, cutoff, dpi):
    views = list(VIEW_CAMERAS)
    fig, axes = plt.subplots(len(LANDMARKS),len(views),figsize=(3.5*len(views),9),sharex=True,squeeze=False)
    for col, view in enumerate(views):
        for row, (landmark, label) in enumerate(LANDMARKS.items()):
            ax = axes[row,col]
            part = data[(data.view==view)&(data.landmark==landmark)]
            if part.empty:
                ax.set_axis_off()
                ax.text(.5,.5,'Not labeled',ha='center',va='center',transform=ax.transAxes,color='#777777')
                continue
            record = summary[(summary.view==view)&(summary.landmark==landmark)].iloc[0]
            ax.set_title(f'View {view} · {label}',loc='left',fontsize=10,weight='bold')
            if kind=='displacement':
                ax.plot(part.frame_index,part.displacement_px,color=COLORS['original'],lw=1.6)
                ax.set_ylim(bottom=0)
                ax.set_ylabel('2D shift (px)')
                ax.text(.03,.94,f'Mean {record.mean_displacement_px:.2f}\nMax {record.max_displacement_px:.2f}',transform=ax.transAxes,va='top',fontsize=8,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
            else:
                ax.plot(part.frame_index,part.original_likelihood,color=COLORS['original'],lw=1.5)
                ax.plot(part.frame_index,part.edited_likelihood,color=COLORS['edited'],ls='--',lw=1.5)
                ax.axhline(cutoff,color='#888888',ls=':',lw=.8)
                ax.set_ylim(-.035,1.035)
                ax.set_ylabel('DLC likelihood')
            format_axis(ax)
            if row==len(LANDMARKS)-1:
                ax.set_xlabel('Frame #')
    if kind=='likelihood':
        legend(fig,cutoff)
    fig.tight_layout(rect=(0,0,1,.94 if kind=='likelihood' else 1),h_pad=1.2)
    save_figure(fig,output,f'{kind}_overview',dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input',type=Path,help='Folder (searched recursively) or ZIP containing paired raw DLC CSVs.')
    parser.add_argument('--output',type=Path,default=Path('marker_analysis'))
    parser.add_argument('--likelihood-cutoff',type=float,default=.8,help='Diagnostic line/count only; never filters coordinates.')
    parser.add_argument('--dpi',type=int,default=200,help='PNG resolution; PDFs retain vector lines.')
    args = parser.parse_args()
    if not 0<=args.likelihood_cutoff<=1 or args.dpi<50:
        parser.error('Likelihood cutoff must be 0-1 and dpi must be at least 50.')
    try:
        tables, info, metadata = load_inputs(args.input)
        for camera in VIEW_CAMERAS.values():
            for field in ['fps','nframes','batch_size','frame_dimensions','cropping']:
                a,b = metadata.get((camera,'markers'),{}).get(field),metadata.get((camera,'no_markers'),{}).get(field)
                if a is not None and b is not None and a!=b:
                    raise ValueError(f'Camera {camera}: metadata mismatch for {field}: {a} versus {b}.')
            for condition in ['markers','no_markers']:
                count = metadata.get((camera,condition),{}).get('nframes')
                if count is not None and count!=len(tables[(camera,condition)]):
                    raise ValueError(f'Camera {camera}, {condition}: metadata frame count disagrees with CSV.')
        summary,data = compare_views(tables,args.likelihood_cutoff)
    except (ValueError,KeyError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output/'landmark_summary.csv',index=False)
    data.to_csv(args.output/'paired_frame_predictions.csv',index=False)
    data.nlargest(20,'displacement_px').to_csv(args.output/'largest_prediction_shifts.csv',index=False)
    manifest = {'source':str(args.input),'view_to_camera':VIEW_CAMERAS,'excluded_landmarks':{str(k):sorted(v) for k,v in EXCLUDED.items()},
                'exclusion_reason':'User-specified annotation coverage, applied to both conditions for all frames.',
                'inputs':[{'camera':cam,'condition':condition,**record,'metadata':metadata.get((cam,condition),{})} for (cam,condition),record in sorted(info.items())],
                'method':'Raw CSVs. Exact matching frame indices. No smoothing, alignment, rescaling, or likelihood filtering.',
                'limitation':'Matching metadata and CSV indices do not independently verify source-frame identity or training exclusion.'}
    (args.output/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.facecolor':'white'})
    plot_xy(data,args.output,args.dpi)
    for kind in ['displacement','likelihood']:
        plot_overview(data,summary,args.output,kind,args.likelihood_cutoff,args.dpi)
    print(summary[['view','landmark','frames','mean_displacement_px','max_displacement_px']].round(3).to_string(index=False))
    print(f'\nSaved plots, data, and provenance to: {args.output.resolve()}')


if __name__=='__main__':
    main()
