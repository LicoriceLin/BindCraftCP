from ._import import *
from typing import  Sequence
from seaborn.palettes import _ColorPalette

xkcd_color=lambda x:mcolors.to_rgb(mcolors.XKCD_COLORS[f'xkcd:{x}'])
def configure_rcParams():
    c_rcParams=plt.rcParamsDefault
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{amsmath}",
        'svg.fonttype':'none',
        'font.sans-serif':['Arial','Helvetica',
            'DejaVu Sans',
            'Bitstream Vera Sans',
            'Computer Modern Sans Serif',
            'Lucida Grande',
            'Verdana',
            'Geneva',
            'Lucid',
            'Avant Garde',
            'sans-serif'],
        "pdf.use14corefonts":False,
        'pdf.fonttype':42,
        'text.color':xkcd_color('dark grey'),
        'axes.labelweight':'heavy',
        'axes.titleweight':'extra bold',
        'figure.facecolor':'none',
        'savefig.transparent':True,
            })
    return c_rcParams

def plot_protein_features(
        seq:str, features:List[np.ndarray], 
        feature_names:List[str],colors:List[str|tuple],
        chunk_size:int=30,
        width:float=10.,height_single:float=1.5,
        exclude_annot:List[str]=[]):
    L = len(seq)
    N = len(features)
    # chunk_size = 30
    num_rows = math.ceil(L / chunk_size)
    
    fig, axes = plt.subplots(num_rows*(N + 2),1, figsize=(width, num_rows * height_single),sharex=True)

    if num_rows == 1:
        axes = [axes]
    axes:List[plt.Axes]
    

    def to_annot(feature:np.ndarray,threshold:float=0.5):
        o=[]
        for i in feature.reshape(-1).tolist():
            if i>threshold:
                o.append(f'{int(i*100)}')
            else:
                o.append('')
        return o
    for row in range(num_rows):
        start = row * chunk_size
        end = min((row + 1) * chunk_size, L)
        
        for i, feature in enumerate(features):
            data = feature[start:end].reshape(1, -1)
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["white", colors[i]])
            axes[row*(N+2)+i].imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1,alpha=0.7)
            axes[row*(N+2)+i].set_xticks([])
            axes[row*(N+2)+i].set_yticks([])
            axes[row*(N+2)+i].set_ylabel(feature_names[i], fontsize=6, rotation=0, labelpad=10, va="center")
            if feature_names[i] not in exclude_annot:
                threshold =0.5 if feature_names[i] != 'sasa' else 0.4
                for j, letter in enumerate(to_annot(data,threshold)):
                    axes[row*(N+2)+i].text(j, 0, letter, ha="center", va="center", fontsize=6, fontweight="bold",color=colors[i])
            # axes[row*(N+2)+i]
        # Residue sequence track
        axes[row*(N+2)+N].imshow(np.zeros((1, end - start)), aspect="auto", cmap="Greys")
        axes[row*(N+2)+N].spines['bottom'].set_visible(False)
        axes[row*(N+2)+N].set_xticks([])
        axes[row*(N+2)+N].set_yticks([])
        
        for j, letter in enumerate(seq[start:end]):
            axes[row*(N+2)+N].text(j, 0, letter, ha="center", va="center", fontsize=6, fontweight="bold")
        
        # Residue index
        axes[row*(N+2)+N].set_ylabel(f"{start + 1}-{end}", fontsize=6, rotation=0, labelpad=10, va="center")
        axes[row*(N+2)+N+1].set_axis_off()
    plt.subplots_adjust(wspace=0, hspace=0)

    for ax in axes:
        ax.grid('off')
        ax.margins(x=0.)
        ax.yaxis.get_label().set_horizontalalignment('right')
        for i in ['left','right']: #'top','bottom',
            ax.spines[i].set_visible(False)
    return fig, axes

def kde_scatter(df:pd.DataFrame,x:str,y:str,hue:str|None=None,color='tab:blue',ax:plt.Axes|None=None):
    if ax is None:
        fig,ax=plt.subplots(1,1)
    else:
        fig=ax.get_figure()
    sns.kdeplot(data=df, x=x,y=y,hue=hue,fill=True,alpha=0.4,color=color,ax=ax)
    sns.scatterplot(data=df,x=x,y=y,hue=hue,ax=ax,color=color)
    return fig,ax

def joint_kde_scatter(df:pd.DataFrame,x:str,y:str,hue:str|None=None,color='tab:blue'):
    g = sns.jointplot(data=df, x=x, y=y,hue=hue,color=color)
    g.plot_joint(sns.kdeplot, color=color, hue=hue,fill=True,alpha=0.4)
    return g

def sort_categories(df:pd.DataFrame,col:str,orders:List[str]):
    '''
    order label cols
    for sns plots.
    '''
    dt = CategoricalDtype(categories=orders, ordered=True)
    df[col] = df[col].astype(dt)
    return df

def strip_box(data:pd.DataFrame,x:str,y:str,ax:plt.Axes,hue:str|None=None,
    order:Sequence[str]|None=None,hue_order:Sequence[str]|None=None,
    palette:_ColorPalette|None=None
    ):
    sns.boxplot(data,y=y,x=x,hue=hue, fliersize=0,boxprops={'alpha': 0.4},order=order,ax=ax,palette=palette)
    sns.stripplot(data,y=y,x=x,ax=ax,hue=hue,order=order,dodge=True,palette=palette)
    if hue!=None:
        handles, labels = ax.get_legend_handles_labels()
        r_=len(data[hue].unique())
        ax.legend(handles=[(handles[i], handles[i+r_]) for i in range(r_)],
            labels=hue_order)
    return ax

def add_annot(pairs:List[Any],data:pd.DataFrame,x:str,y:str,ax:plt.Axes,hue:str|None=None,
    order:Sequence[str]|None=None,hue_order:Sequence[str]|None=None):
    annotator = Annotator(ax, pairs,hue=hue,data=data, x=x, y=y, order=order,hue_order=hue_order,plot='boxplot')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    return ax

def add_filter_bar(ax:plt.Axes,filter_dict:Dict[str,bool|float|int|None]):
    c='tab:red' if filter_dict['higher'] else 'green'
    y_thresh=filter_dict['threshold']
    xmin,xmax=ax.get_xlim()
    ymin,ymax=ax.get_ylim()
    arrow_x=(xmin+xmax)/2
    dy=(ymax-ymin)*0.05 if filter_dict['higher'] else -(ymax-ymin)*0.05
    ax.hlines(y_thresh,xmin,xmax,colors=c,linestyles='--')
    ax.arrow(arrow_x,y_thresh,0,dy,color=c,head_width=0.05,head_length=abs(dy*0.5),length_includes_head=True)
    return ax