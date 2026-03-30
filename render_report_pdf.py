import json
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML


def render_pdf(report_json, template_dir, out_pdf):
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html")

    html_content = template.render(**data)

    HTML(string=html_content).write_pdf(out_pdf)
    print(f"[OK] PDF generated: {out_pdf}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--template_dir", default="templates")
    ap.add_argument("--out", default="report.pdf")
    args = ap.parse_args()

    render_pdf(args.report, args.template_dir, args.out)