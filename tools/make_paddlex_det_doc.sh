#!/bin/bash

PROJECT="paddlex_det"
DOC_TITLE="paddlex_det 参数说明"
DOC_FILE="docs/paddlex_det_doc.md"

make_doc() {
    app_name=$1
    echo "## ${app_name}.py">>"$DOC_FILE"
    echo "">>"$DOC_FILE"
    echo "\`\`\`bash">>"$DOC_FILE"
    python3 run/${app_name}.py -h>>"$DOC_FILE"
    echo "\`\`\`">>"$DOC_FILE"
    echo "">>"$DOC_FILE"
}

make_project_addr() {
    echo "## 项目地址">>"$DOC_FILE"
    echo "">>"$DOC_FILE"
    echo "- [gitee](https://gitee.com/cnhemiya/bmm-paddle-helper)">>"$DOC_FILE"
    echo "- [github](https://github.com/cnhemiya/bmm-paddle-helper)">>"$DOC_FILE"
    echo "">>"$DOC_FILE"
}

python3 tools/mkbmmph.py --project $PROJECT

echo "## ${DOC_TITLE}">"$DOC_FILE"
echo "">>"$DOC_FILE"
make_doc train
make_doc quant
make_doc prune
make_doc infer
make_project_addr
