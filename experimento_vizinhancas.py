#!/usr/bin/env python3
"""Experimento da T2: comparação estatística das vizinhanças do Greedy Descent.

Uso:
    python experimento_vizinhancas.py manutencao_pequena.txt manutencao_media.txt manutencao_grande.txt

O script grava quatro CSVs: resultados brutos, resumo descritivo, testes estatísticos
(Friedman + Wilcoxon pareado com correção de Holm) e benchmark de custo por movimento.

Por padrão, usa Best Improvement para isolar o efeito da vizinhança. Os mesmos seeds
são usados nas três vizinhanças de cada instância; assim, a primeira permutação aleatória
é igual entre tratamentos, e a comparação é analisada como pareada.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

from problema import ProblemaManutencao, VIZINHANCAS
from busca_local import greedy_descent

try:
    from scipy.stats import friedmanchisquare, wilcoxon
except ImportError:  # Permite que o experimento rode sem SciPy; estatística fica indisponível.
    friedmanchisquare = None
    wilcoxon = None


ORCAMENTOS_POR_N = ((12, 50_000), (50, 150_000), (150, 300_000))
REPETICOES_POR_N = ((12, 30), (50, 30), (150, 10))
ALPHA = 0.05


def valor_por_n(n: int, tabela, nome: str) -> int:
    for limite, valor in tabela:
        if n <= limite:
            return valor
    raise ValueError(
        f"n={n} não tem {nome} padrão configurado. Informe --orcamento e/ou --repeticoes."
    )


def gravar_csv(caminho: Path, linhas: list[dict]) -> None:
    if not linhas:
        return
    # Algumas linhas (p.ex., testes pareados) têm colunas adicionais.
    campos = list(dict.fromkeys(chave for linha in linhas for chave in linha.keys()))
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)


def media_segura(valores):
    return statistics.mean(valores) if valores else math.nan


def desvio_seguro(valores):
    return statistics.stdev(valores) if len(valores) > 1 else 0.0


def holm(p_valores: list[float]) -> list[float]:
    """Retorna p-valores ajustados pelo procedimento step-down de Holm."""
    m = len(p_valores)
    ordenados = sorted(enumerate(p_valores), key=lambda item: item[1])
    ajustados = [0.0] * m
    maior_ajustado_anterior = 0.0
    for pos, (indice_original, p) in enumerate(ordenados):
        ajuste = min(1.0, (m - pos) * p)
        ajuste = max(ajuste, maior_ajustado_anterior)
        ajustados[indice_original] = ajuste
        maior_ajustado_anterior = ajuste
    return ajustados


def benchmark_movimentos(caminho: str, nome_instancia: str, repeticoes: int,
                         seed_base: int) -> list[dict]:
    """Mede o tempo médio da avaliação incremental de cada movimento.

    Cada vizinhança é avaliada em uma permutação aleatória comum por seed. A lista de
    movimentos e os prefixos são preparados antes da cronometragem. O tempo inclui a
    função avaliar_movimento, mas não inclui a criação/shuffle da lista de movimentos.
    """
    linhas = []
    for r in range(repeticoes):
        seed = seed_base + r
        for viz in VIZINHANCAS:
            problema = ProblemaManutencao(caminho, vizinhanca=viz, incremental=True)
            problema.reiniciar(semente=seed, orcamento=None)
            s = problema.solucao_aleatoria()
            h_base = problema.h(s)  # registra o incumbente para permitir a poda incremental
            problema._prefixos(s)   # aquece/prepara o cache antes de medir
            movimentos = list(problema._movimentos)

            inicio = time.perf_counter()
            for mov in movimentos:
                # O limite h_base reproduz a pergunta comum: este vizinho melhora a solução?
                problema.avaliar_movimento(s, mov, limite=h_base)
            duracao = time.perf_counter() - inicio
            n_mov = len(movimentos)

            linhas.append({
                "instancia": nome_instancia,
                "n": problema.n,
                "vizinhanca": viz,
                "seed": seed,
                "h_solucao_base": h_base,
                "movimentos_avaliados": n_mov,
                "tempo_total_s": duracao,
                "tempo_medio_ms_por_movimento": 1000.0 * duracao / n_mov,
                "avaliacao_incremental": True,
            })
    return linhas


def calcular_testes_por_instancia(instancia: str, n: int, estrategia: str,
                                  dados: list[dict], campo: str,
                                  nome_metrica: str) -> list[dict]:
    """Testes para três condições pareadas: Friedman e Wilcoxon pareado + Holm."""
    if friedmanchisquare is None or wilcoxon is None:
        return [{
            "instancia": instancia, "n": n, "estrategia": estrategia,
            "metrica": nome_metrica, "teste": "indisponivel",
            "comparacao": "",
            "estatistica": "", "p_valor": "", "p_valor_ajustado_holm": "",
            "alpha": ALPHA, "significativo": "",
            "observacao": "Instale scipy (python -m pip install scipy) para os testes.",
        }]

    por_viz_seed = defaultdict(dict)
    for row in dados:
        raw = row.get(campo, "")
        try:
            valor = float(raw)
        except (TypeError, ValueError):
            valor = math.nan
        por_viz_seed[row["vizinhanca"]][row["seed"]] = valor

    vizinhs = list(VIZINHANCAS)
    seeds_comuns = sorted(set.intersection(*(set(por_viz_seed[v]) for v in vizinhs)))
    # Para métricas de ótimos locais confirmados, exclui apenas os blocos em que
    # alguma vizinhança não chegou a concluir uma varredura completa.
    seeds_comuns = [
        seed for seed in seeds_comuns
        if all(math.isfinite(por_viz_seed[v][seed]) for v in vizinhs)
    ]
    if len(seeds_comuns) < 3:
        return [{
            "instancia": instancia, "n": n, "estrategia": estrategia,
            "metrica": nome_metrica, "teste": "insuficiente",
            "comparacao": "", "estatistica": "", "p_valor": "",
            "p_valor_ajustado_holm": "", "alpha": ALPHA, "significativo": "",
            "observacao": f"Apenas {len(seeds_comuns)} blocos/seeds comuns.",
        }]

    grupos = [[por_viz_seed[v][seed] for seed in seeds_comuns] for v in vizinhs]
    linhas = []
    todos_iguais = all(
        all(valor == grupo[0] for valor in grupo) for grupo in grupos
    ) and all(grupo[0] == grupos[0][0] for grupo in grupos)
    if todos_iguais:
        stat_f, p_f = 0.0, 1.0
        obs_f = "Todos os resultados são iguais; não há variação para testar."
    else:
        try:
            stat_f, p_f = friedmanchisquare(*grupos)
            if not math.isfinite(float(p_f)):
                stat_f, p_f = 0.0, 1.0
                obs_f = "O teste retornou valor não finito; tratado como sem evidência de diferença."
            else:
                obs_f = "Teste global para diferenças entre as três vizinhanças."
        except ValueError as exc:
            stat_f, p_f = math.nan, math.nan
            obs_f = f"Não foi possível calcular Friedman: {exc}"

    linhas.append({
        "instancia": instancia, "n": n, "estrategia": estrategia,
        "metrica": nome_metrica, "teste": "Friedman",
        "comparacao": "swap; insertion; 2opt", "estatistica": stat_f,
        "p_valor": p_f, "p_valor_ajustado_holm": "", "alpha": ALPHA,
        "significativo": bool(p_f < ALPHA) if not math.isnan(p_f) else "",
        "observacao": obs_f,
    })

    pares = [("swap", "insertion"), ("swap", "2opt"), ("insertion", "2opt")]
    p_crus = []
    resultados_pares = []
    for a, b in pares:
        va = [por_viz_seed[a][seed] for seed in seeds_comuns]
        vb = [por_viz_seed[b][seed] for seed in seeds_comuns]
        diferencas = [x - y for x, y in zip(va, vb)]
        try:
            if all(d == 0 for d in diferencas):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(va, vb, alternative="two-sided", zero_method="wilcox")
            obs = "Delta mediano = mediana(A - B); negativo favorece A para custo h."
        except ValueError as exc:
            stat, p = math.nan, 1.0
            obs = f"Teste não calculável; tratado conservadoramente como p=1: {exc}"
        p_crus.append(p)
        resultados_pares.append({
            "instancia": instancia, "n": n, "estrategia": estrategia,
            "metrica": nome_metrica, "teste": "Wilcoxon pareado",
            "comparacao": f"{a} vs {b}", "estatistica": stat,
            "p_valor": p, "p_valor_ajustado_holm": "", "alpha": ALPHA,
            "significativo": "", "observacao": obs,
            "mediana_delta_A_menos_B": statistics.median(diferencas),
            "A_menor_que_B": sum(x < y for x, y in zip(va, vb)),
            "empates": sum(x == y for x, y in zip(va, vb)),
            "A_maior_que_B": sum(x > y for x, y in zip(va, vb)),
        })
    p_ajustados = holm(p_crus)
    for linha, p_adj in zip(resultados_pares, p_ajustados):
        linha["p_valor_ajustado_holm"] = p_adj
        linha["significativo"] = bool(p_adj < ALPHA)
        linhas.append(linha)
    return linhas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instancias", nargs="*",
        help="Caminhos dos arquivos manutencao_*.txt. Se omitidos, tenta os três nomes oficiais.",
    )
    parser.add_argument("--estrategia", choices=("best", "first"), default="best",
                        help="Estratégia mantida fixa ao comparar vizinhanças (padrão: best).")
    parser.add_argument("--orcamento", type=int, default=None,
                        help="Sobrescreve o orçamento padrão de cada instância.")
    parser.add_argument("--repeticoes", type=int, default=None,
                        help="Sobrescreve o número padrão de repetições para todas as instâncias.")
    parser.add_argument("--seed-base", type=int, default=2026,
                        help="Primeira seed; as mesmas seeds são usadas em cada vizinhança.")
    parser.add_argument("--bench-repeticoes", type=int, default=10,
                        help="Número de permutações/seeds no benchmark por movimento.")
    parser.add_argument("--saida", default="resultados_vizinhancas",
                        help="Prefixo dos quatro CSVs de saída (padrão: resultados_vizinhancas).")
    args = parser.parse_args()

    caminhos = args.instancias or [
        "manutencao_pequena.txt", "manutencao_media.txt", "manutencao_grande.txt"
    ]
    for caminho in caminhos:
        if not os.path.isfile(caminho):
            raise FileNotFoundError(
                f"Não encontrei '{caminho}'. Informe o caminho correto do arquivo de instância."
            )
    if args.orcamento is not None and args.orcamento <= 0:
        parser.error("--orcamento deve ser positivo")
    if args.repeticoes is not None and args.repeticoes < 3:
        parser.error("--repeticoes deve ser pelo menos 3 para os testes estatísticos")
    if args.bench_repeticoes < 1:
        parser.error("--bench-repeticoes deve ser positivo")

    dados_brutos = []
    dados_benchmark = []
    for caminho in caminhos:
        nome = Path(caminho).stem
        # Carrega a instância e usa n para selecionar o protocolo padrão.
        p0 = ProblemaManutencao(caminho)
        n = p0.n
        budget = args.orcamento or valor_por_n(n, ORCAMENTOS_POR_N, "orçamento")
        reps = args.repeticoes or valor_por_n(n, REPETICOES_POR_N, "número de repetições")
        print(f"\nInstância {nome}: n={n}, orçamento={budget:,}, repetições={reps}")

        for i in range(reps):
            seed = args.seed_base + i
            for viz in VIZINHANCAS:
                problema = ProblemaManutencao(caminho, vizinhanca=viz, incremental=True)
                inicio = time.perf_counter()
                resultado = greedy_descent(
                    problema,
                    estrategia=args.estrategia,
                    semente=seed,
                    orcamento=budget,
                )
                segundos = time.perf_counter() - inicio
                otimos_completos = resultado["execucoes_com_otimo_local_completo"]
                hs_otimos_completos = [item["h_otimo_local"] for item in otimos_completos]
                hs_otimos_distintos = [item["h"] for item in resultado["otimos_locais_distintos"]]
                dados_brutos.append({
                    "instancia": nome,
                    "caminho_instancia": caminho,
                    "n": n,
                    "estrategia": args.estrategia,
                    "vizinhanca": viz,
                    "seed": seed,
                    "orcamento": budget,
                    "avaliacoes": resultado["avaliacoes"],
                    "melhor_h": resultado["melhor_h"],
                    "melhor_solucao": repr(resultado["melhor_s"]),
                    "tempo_s": segundos,
                    "tempo_ms_por_avaliacao": 1000.0 * segundos / max(1, resultado["avaliacoes"]),
                    "reinicios": resultado["reinicios"],
                    "passos_melhoria": resultado["passos_melhoria"],
                    "otimos_locais_distintos_na_execucao": resultado["quantidade_otimos_locais_distintos"],
                    "otimos_locais_completos_registrados": len(otimos_completos),
                    "melhor_h_otimo_local_confirmado": min(hs_otimos_completos) if hs_otimos_completos else math.nan,
                    "media_h_otimos_locais_por_reinicio": statistics.mean(hs_otimos_completos) if hs_otimos_completos else math.nan,
                    "mediana_h_otimos_locais_por_reinicio": statistics.median(hs_otimos_completos) if hs_otimos_completos else math.nan,
                    "custos_otimos_locais_distintos": repr(sorted(hs_otimos_distintos)),
                })
                print(
                    f"  {viz:10s} seed={seed} h={resultado['melhor_h']:.0f} "
                    f"eval={resultado['avaliacoes']} tempo={segundos:.2f}s"
                )

        print(f"Benchmark de custo por movimento: {args.bench_repeticoes} repetições por vizinhança...")
        dados_benchmark.extend(
            benchmark_movimentos(caminho, nome, args.bench_repeticoes, args.seed_base + 100_000)
        )

    # Estatística descritiva da qualidade e do custo de execução do Greedy Descent.
    agrupado = defaultdict(list)
    for row in dados_brutos:
        agrupado[(row["instancia"], row["n"], row["estrategia"], row["vizinhanca"])].append(row)
    resumo = []
    for (inst, n, estrategia, viz), rows in agrupado.items():
        hs = [float(r["melhor_h"]) for r in rows]
        tempos = [float(r["tempo_s"]) for r in rows]
        por_eval = [float(r["tempo_ms_por_avaliacao"]) for r in rows]
        resumo.append({
            "instancia": inst, "n": n, "estrategia": estrategia,
            "vizinhanca": viz, "repeticoes": len(rows),
            "media_h": statistics.mean(hs), "desvio_padrao_h": desvio_seguro(hs),
            "mediana_h": statistics.median(hs), "minimo_h": min(hs), "maximo_h": max(hs),
            "media_tempo_s": statistics.mean(tempos), "mediana_tempo_s": statistics.median(tempos),
            "media_ms_por_avaliacao_execucao": statistics.mean(por_eval),
            "media_reinicios": media_segura([r["reinicios"] for r in rows]),
            "media_passos_melhoria": media_segura([r["passos_melhoria"] for r in rows]),
            "media_otimos_locais_distintos": media_segura([r["otimos_locais_distintos_na_execucao"] for r in rows]),
            "media_melhor_h_otimo_local_confirmado": media_segura([
                float(r["melhor_h_otimo_local_confirmado"]) for r in rows
                if math.isfinite(float(r["melhor_h_otimo_local_confirmado"]))
            ]),
            "mediana_melhor_h_otimo_local_confirmado": statistics.median([
                float(r["melhor_h_otimo_local_confirmado"]) for r in rows
                if math.isfinite(float(r["melhor_h_otimo_local_confirmado"]))
            ]) if any(math.isfinite(float(r["melhor_h_otimo_local_confirmado"])) for r in rows) else math.nan,
        })

    testes = []
    instancias_no_dado = sorted({(r["instancia"], r["n"]) for r in dados_brutos})
    for inst, n in instancias_no_dado:
        subset = [r for r in dados_brutos if r["instancia"] == inst]
        testes.extend(calcular_testes_por_instancia(
            inst, n, args.estrategia, subset, "melhor_h", "melhor_h"
        ))
        testes.extend(calcular_testes_por_instancia(
            inst, n, args.estrategia, subset,
            "melhor_h_otimo_local_confirmado", "melhor_h_otimo_local_confirmado"
        ))

    # Resumo do benchmark e testes pareados do tempo por movimento.
    benchmark_resumo = []
    for (inst, n, viz), rows in _agrupa(dados_benchmark, ("instancia", "n", "vizinhanca")).items():
        ts = [float(r["tempo_medio_ms_por_movimento"]) for r in rows]
        benchmark_resumo.append({
            "instancia": inst, "n": n, "vizinhanca": viz, "repeticoes": len(rows),
            "movimentos_por_teste": rows[0]["movimentos_avaliados"],
            "media_ms_por_movimento": statistics.mean(ts),
            "desvio_padrao_ms_por_movimento": desvio_seguro(ts),
            "mediana_ms_por_movimento": statistics.median(ts),
            "minimo_ms_por_movimento": min(ts), "maximo_ms_por_movimento": max(ts),
        })
    for inst, n in instancias_no_dado:
        subset = [r for r in dados_benchmark if r["instancia"] == inst]
        testes.extend(calcular_testes_por_instancia(
            inst, n, args.estrategia, subset, "tempo_medio_ms_por_movimento",
            "tempo_ms_por_movimento"
        ))

    prefixo = Path(args.saida)
    saidas = {
        "brutos": Path(f"{prefixo}_brutos.csv"),
        "resumo": Path(f"{prefixo}_resumo.csv"),
        "testes": Path(f"{prefixo}_testes.csv"),
        "benchmark_bruto": Path(f"{prefixo}_benchmark_bruto.csv"),
        "benchmark_resumo": Path(f"{prefixo}_benchmark_resumo.csv"),
    }
    gravar_csv(saidas["brutos"], dados_brutos)
    gravar_csv(saidas["resumo"], resumo)
    gravar_csv(saidas["testes"], testes)
    gravar_csv(saidas["benchmark_bruto"], dados_benchmark)
    gravar_csv(saidas["benchmark_resumo"], benchmark_resumo)

    print("\nArquivos gerados:")
    for caminho in saidas.values():
        print(f"  {caminho}")
    if friedmanchisquare is None or wilcoxon is None:
        print("\nOBS.: scipy não está instalado; instale com: python -m pip install scipy")
    print("\nExecução concluída. Confira primeiro os CSVs de resumo e testes.")


def _agrupa(linhas, campos):
    grupos = defaultdict(list)
    for linha in linhas:
        grupos[tuple(linha[c] for c in campos)].append(linha)
    return grupos


if __name__ == "__main__":
    main()
