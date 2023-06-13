import seaborn as sns
from spacy import displacy


def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def make_color_palette(labels):
    color_palette = sns.color_palette(n_colors=len(labels))
    return {x: rgb2hex(*y) for x, y in zip(labels, color_palette)}


def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


def visualize_ner(text, rlt, colors, span=False):
    if span:
        spans = [{"start_token": t["start"], "end_token": t["end"], "label": k} for k, v in rlt.items() for t in v]

        ex = [{"text": text, "spans": spans, "tokens": list(text)}]
        return displacy.render(ex, style="span", manual=True, options={"colors": colors})

    else:
        ents = [{"start": t["start"], "end": t["end"], "label": k} for k, v in rlt.items() for t in v]

        ex = [{"text": text, "tokens": list(text), "ents": ents}]
        return displacy.render(ex, style="ent", manual=True, options={"colors": colors})
