import prelude

import os
import sys
import json
import torch
from datetime import datetime, timezone
from model import Brain, DQN, GRP
from engine import MortalEngine
from common import filtered_trimmed_lines
from libriichi.mjai import Bot
from libriichi.dataset import Grp
from config import config

USAGE = '''Usage: python mortal.py <ID>

ARGS:
    <ID>    The player ID, an integer within [0, 3].'''


# === [추가] 프리프로세스 스위치 판단 ===
def _is_preprocess_mode(argv):
    # 1) 환경변수 MORTAL_MODE=preprocess
    if os.environ.get("MORTAL_MODE", "").lower() == "preprocess":
        return True, argv[1:]   # <<< 인자 전부 넘기기
    # 2) CLI 플래그: --preprocess (그 뒤의 인자들은 preprocess로 넘김)
    if len(argv) > 1 and argv[1] in ("--preprocess", "preprocess"):
        return True, argv[2:]
    return False, []

# === [추가] 프리프로세스 진입 ===
def _run_preprocess(rest_args):
    try:
        # preprocess.py가 mortal/ 폴더 안에 있어야 합니다. (이미 COPY mortal/ ./ 로 들어오므로 OK)
        import preprocess
    except Exception as e:
        print(f"[fatal] preprocess 모듈을 불러올 수 없습니다: {e}", file=sys.stderr)
        sys.exit(2)

    # preprocess.main()이 argparse를 사용한다면, sys.argv를 재구성해 호출하는 방법이 가장 간단
    # ex) python preprocess.py --in ... --out ... --players 0,1,2,3
    saved_argv = sys.argv[:]
    try:
        sys.argv = ["preprocess.py", *rest_args]
        preprocess.main()
    finally:
        sys.argv = saved_argv
    # preprocess 실행 후 종료
    sys.exit(0)

def main():
    # === [추가] 모드 스위치 체크 ===
    is_pp, rest = _is_preprocess_mode(sys.argv)
    if is_pp:
        _run_preprocess(rest)

    # --- 이하 기존 mortal.py 로직 그대로 ---
    try:
        player_id = int(sys.argv[-1])
        assert player_id in range(4)
    except:
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    review_mode = os.environ.get('MORTAL_REVIEW_MODE', '0') == '1'

    device = torch.device('cpu')
    state = torch.load(config['control']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks']
    conv_channels = cfg['resnet']['conv_channels']
    if 'tag' in state:
        tag = state['tag']
    else:
        time = datetime.fromtimestamp(state['timestamp'], tz=timezone.utc).strftime('%y%m%d%H')
        tag = f'mortal{version}-b{num_blocks}c{conv_channels}-t{time}'

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    engine = MortalEngine(
        mortal,
        dqn,
        version = version,
        is_oracle = False,
        device = device,
        enable_amp = False,
        enable_quick_eval = not review_mode,
        enable_rule_based_agari_guard = True,
        name = 'mortal',
    )
    bot = Bot(engine, player_id)

    if review_mode:
        logs = []

    for line in filtered_trimmed_lines(sys.stdin):
        if review_mode:
            logs.append(line)

        if reaction := bot.react(line):
            print(reaction, flush=True)
        elif review_mode:
            print('{"type":"none","meta":{"mask_bits":0}}', flush=True)

    # --- 리뷰모드 후처리 (GRP 없음 시 스킵) ---
    if review_mode:
        try:
            # GRP 관련 설정이 없으면 그냥 스킵
            if 'grp' not in config:
                print(json.dumps({
                    "model_tag": tag,
                    "note": "GRP section missing; skipping phi_matrix analysis"
                }), flush=True)
                return

            from libriichi.dataset import Grp
            grp = GRP(**config['grp']['network'])
            grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cuda'))
            grp.load_state_dict(grp_state['model'])

            ins = Grp.load_log('\n'.join(logs))
            feature = ins.take_feature()
            seq = [torch.as_tensor(feature[:idx+1], device=device) for idx in range(len(feature))]

            with torch.inference_mode():
                logits = grp(seq)
            matrix = grp.calc_matrix(logits)
            extra_data = {
                "model_tag": tag,
                "phi_matrix": matrix.tolist(),
            }
            print(json.dumps(extra_data), flush=True)
        except Exception as e:
            print(json.dumps({
                "model_tag": tag,
                "note": f"GRP analysis skipped due to error: {str(e)}"
            }), flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

