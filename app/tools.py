import ast
import operator
import re

import requests


TOOL_CALL_LOG = []


def reset_tool_log():
    TOOL_CALL_LOG.clear()


def get_tool_log():
    return TOOL_CALL_LOG.copy()


# -----------------------------
# Calculator Tool
# -----------------------------

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate_node(node):
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("숫자만 사용할 수 있습니다.")

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("지원하지 않는 연산입니다.")

        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        # 너무 큰 거듭제곱 방지
        if operator_type is ast.Pow and abs(right) > 20:
            raise ValueError("지수가 너무 큽니다.")

        return ALLOWED_OPERATORS[operator_type](
            left,
            right
        )

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("지원하지 않는 연산입니다.")

        value = _evaluate_node(node.operand)

        return ALLOWED_OPERATORS[operator_type](
            value
        )

    raise ValueError(
        "허용되지 않은 계산식입니다."
    )


def calculate_expression(expression: str) -> dict:
    """
    수학 계산식을 안전하게 계산합니다.

    Args:
        expression:
            계산할 수식.
            예: (10 + 5) * 3, 2 ** 8

    Returns:
        계산식과 계산 결과.
    """

    try:
        tree = ast.parse(
            expression,
            mode="eval"
        )

        result = _evaluate_node(tree)

        response = {
            "expression": expression,
            "result": result,
        }

        TOOL_CALL_LOG.append(
            {
                "tool": "calculate_expression",
                "input": expression,
                "result": result,
            }
        )

        return response

    except Exception as e:
        response = {
            "expression": expression,
            "error": str(e),
        }

        TOOL_CALL_LOG.append(
            {
                "tool": "calculate_expression",
                "input": expression,
                "error": str(e),
            }
        )

        return response


# -----------------------------
# NVD CVE API Tool
# -----------------------------

def lookup_cve(cve_id: str) -> dict:
    """
    NIST NVD API에서 CVE 정보를 조회합니다.

    Args:
        cve_id:
            조회할 CVE ID.
            예: CVE-2021-44228

    Returns:
        CVE 설명, CVSS 점수, 심각도,
        게시일 등의 정보.
    """

    cve_id = cve_id.strip().upper()

    pattern = r"^CVE-\d{4}-\d{4,}$"

    if not re.match(pattern, cve_id):
        result = {
            "error": "올바른 CVE ID 형식이 아닙니다.",
            "example": "CVE-2021-44228",
        }

        TOOL_CALL_LOG.append(
            {
                "tool": "lookup_cve",
                "input": cve_id,
                "error": result["error"],
            }
        )

        return result

    url = (
        "https://services.nvd.nist.gov/"
        "rest/json/cves/2.0"
    )

    try:
        response = requests.get(
            url,
            params={
                "cveId": cve_id
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        vulnerabilities = data.get(
            "vulnerabilities",
            []
        )

        if not vulnerabilities:
            result = {
                "cve_id": cve_id,
                "error": "NVD에서 해당 CVE를 찾지 못했습니다."
            }

            TOOL_CALL_LOG.append(
                {
                    "tool": "lookup_cve",
                    "input": cve_id,
                    "error": result["error"],
                }
            )

            return result

        cve = vulnerabilities[0]["cve"]

        # 영어 설명 추출
        description = None

        for item in cve.get(
            "descriptions",
            []
        ):
            if item.get("lang") == "en":
                description = item.get(
                    "value"
                )
                break

        # CVSS 정보 추출
        metrics = cve.get(
            "metrics",
            {}
        )

        score = None
        severity = None
        cvss_version = None

        metric_keys = [
            "cvssMetricV40",
            "cvssMetricV31",
            "cvssMetricV30",
            "cvssMetricV2",
        ]

        for metric_key in metric_keys:
            metric_list = metrics.get(
                metric_key
            )

            if not metric_list:
                continue

            metric = metric_list[0]

            cvss_data = metric.get(
                "cvssData",
                {}
            )

            score = cvss_data.get(
                "baseScore"
            )

            severity = (
                cvss_data.get(
                    "baseSeverity"
                )
                or metric.get(
                    "baseSeverity"
                )
            )

            cvss_version = cvss_data.get(
                "version"
            )

            break

        references = []

        for reference in cve.get(
            "references",
            []
        )[:5]:
            references.append(
                reference.get("url")
            )

        result = {
            "cve_id": cve.get(
                "id"
            ),
            "description": description,
            "published": cve.get(
                "published"
            ),
            "last_modified": cve.get(
                "lastModified"
            ),
            "cvss_version": cvss_version,
            "cvss_score": score,
            "severity": severity,
            "references": references,
            "source": "NIST NVD",
        }

        TOOL_CALL_LOG.append(
            {
                "tool": "lookup_cve",
                "input": cve_id,
                "result": {
                    "cvss_score": score,
                    "severity": severity,
                }
            }
        )

        return result

    except requests.RequestException as e:
        result = {
            "cve_id": cve_id,
            "error": (
                "NVD API 호출 실패: "
                + str(e)
            )
        }

        TOOL_CALL_LOG.append(
            {
                "tool": "lookup_cve",
                "input": cve_id,
                "error": result["error"],
            }
        )

        return result