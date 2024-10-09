import math
import re


# only support float answer verification
def verify_float(answer: float, output: str):
    if not output:
        print(f'输出为空，无法匹配答案!\n')
        return False

    if '综上所述，' in output:
        spl_ans = output.split('综上所述，')[-1]
        spl_ans = spl_ans.strip()
    else:
        spl_ans = output.strip()

    try:
        match = re.findall(r'-?[0-9]+\.?[0-9]*', spl_ans)[-1]
        model_ans = float(match)

        # standard (adjustable)
        if abs(answer) >= 1:
            result = math.isclose(model_ans, answer, abs_tol=0.1)
        else:
            result = math.isclose(model_ans, answer, rel_tol=0.1)

        print(f'The ans of model is:{model_ans}, while the ground truth is{answer}.\n')
        return result
    except Exception as e:
        print(f'匹配答案出错！错误类型:{e}\n')
        print(f'The ans of model is:{spl_ans}, while the ground truth is{answer}.\n')
        return False


# only support choice answer verification
def verify_choice(answer: str, output: str):
    if not output:
        print(f'输出为空，无法匹配答案!\n')
        return False

    check_list = ['A', 'B', 'C', 'D', 'E']

    if '综上所述，最终答案是:' in output:
        spl_ans = output.split('综上所述，最终答案是:')[-1]
        spl_ans = spl_ans.strip()
    elif '综上所述，' in output:
        spl_ans = output.split('综上所述，')[-1]
        spl_ans = spl_ans.strip()
    else:
        spl_ans = output.strip()

    # standard (adjustable)
    for choice in check_list:
        if choice in answer and choice not in spl_ans:
            print(f'The ans of model is:{spl_ans}, while the ground truth is{answer}.\n')
            return False
        if choice not in answer and choice in spl_ans:
            print(f'The ans of model is:{spl_ans}, while the ground truth is{answer}.\n')
            return False

    print(f'The ans of model is:{spl_ans}, while the ground truth is{answer}.\n')
    return True
