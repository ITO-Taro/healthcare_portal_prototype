from matplotlib.figure import Figure
import io, base64

class Charts:

    def chart_hist(data, bin, title):
        fig = Figure(figsize=(11, 6))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=title, fontdict={'color':'black', 'fontsize':20})
        ax.hist(data, bins=bin)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        url = base64.b64encode(buf.getbuffer()).decode("ascii")
        res = f"img src=data:image/png;base64,{url}"
        return res

    def chart_cumulative(data, bin, title):
        fig = Figure(figsize=(11, 6))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=title, fontdict={'color':'black', 'fontsize':20})
        ax.hist(data, bins=bin, cumulative=True, histtype="step")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        url = base64.b64encode(buf.getbuffer()).decode("ascii")
        res = f"img src=data:image/png;base64,{url}"
        return res