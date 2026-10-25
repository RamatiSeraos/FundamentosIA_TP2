from problema import OrcamentoEsgotado, VIZINHANCAS

ESTRATEGIAS_GREEDY = ("best", "first")


def greedy_descent(problema, estrategia="best", semente=None, orcamento=50_000):
    """
    Parametros
    ---------
    problema : ProblemaManutencao
        Instancia do problema ja configurada com uma vizinhanca.
    estrategia : {"best", "first"}
        "best": avalia toda a vizinhanca e aplica a melhor melhora encontrada.
        "first": percorre os movimentos em ordem aleatoria e aplica a primeira
                 melhora encontrada.
    semente : int | None
        Semente para tornar a execucao reproduzivel.
    orcamento : int
        Numero maximo de avaliacoes da funcao objetivo. Deve ser positivo.

    Retorno
    -------
    dict
        "estrategia"
        "vizinhanca"
        "melhor_s"
        "melhor_h"
        "avaliacoes"
        "orcamento" 
        "reinicios"
        "passos_melhoria"
        "otimos_locais_distintos"
        "quantidade_otimos_locais_distintos" 
        "execucoes_com_otimo_local_completo"
        "historico"
    """
    if estrategia not in ESTRATEGIAS_GREEDY:
        raise ValueError(
            f"Estrategia desconhecida: {estrategia!r}. "
            f"Use uma destas: {ESTRATEGIAS_GREEDY}."
        )
    if not isinstance(orcamento, int) or isinstance(orcamento, bool) or orcamento <= 0:
        raise ValueError("orcamento deve ser um inteiro positivo")
    if problema.n < 2:
        raise ValueError("Greedy Descent requer pelo menos dois ativos")

    problema.reiniciar(semente=semente, orcamento=orcamento)

    reinicios = 0
    passos_melhoria = 0
    execucoes = []
    # Chave: permutacao (tupla). Assim contamos otimos locais distintos.
    otimos_locais_distintos = {}

    try:
        while True:
            # Random restart: comeca de uma nova permutacao aleatoria.
            s = problema.solucao_aleatoria()
            h_atual = problema.h(s)  # conta uma avaliacao e registra a solucao
            reinicios += 1
            h_inicial = h_atual
            passos_nesta_busca = 0

            while True:
                movimento_escolhido = None
                melhor_valor = h_atual

                # movimentos() devolve os movimentos em ordem aleatoria.
                for mov in problema.movimentos():
                    # Basta saber se o vizinho melhora a solucao atual.
                    # Se limite for atingido, a avaliacao incremental pode parar
                    # cedo, pois esse vizinho nao sera aceito.
                    h_vizinho = problema.avaliar_movimento(
                        s, mov, limite=h_atual
                    )

                    if h_vizinho < melhor_valor:
                        melhor_valor = h_vizinho
                        movimento_escolhido = mov

                        if estrategia == "first":
                            break

                if movimento_escolhido is None:
                    # Toda a vizinhanca foi examinada sem melhora: otimo local.
                    chave = tuple(s)
                    otimos_locais_distintos.setdefault(chave, h_atual)
                    execucoes.append({
                        "h_inicial": h_inicial,
                        "h_otimo_local": h_atual,
                        "solucao_otimo_local": list(s),
                        "passos_melhoria": passos_nesta_busca,
                    })
                    break  # inicia outro random restart

                # Aplica a melhora selecionada; o custo ja foi calculado acima.
                s = problema.aplicar(s, movimento_escolhido)
                h_atual = melhor_valor
                passos_nesta_busca += 1
                passos_melhoria += 1

    except OrcamentoEsgotado:
        # A classe lancou a excecao ao tentar ultrapassar o limite. Nenhuma
        # avaliacao extra foi contabilizada: problema.avaliacoes == orcamento.
        pass

    otimos = [
        {"solucao": list(solucao), "h": valor}
        for solucao, valor in otimos_locais_distintos.items()
    ]

    return {
        "estrategia": estrategia,
        "vizinhanca": problema.vizinhanca,
        "melhor_s": list(problema.melhor_s) if problema.melhor_s is not None else None,
        "melhor_h": problema.melhor_h,
        "avaliacoes": problema.avaliacoes,
        "orcamento": orcamento,
        "reinicios": reinicios,
        "passos_melhoria": passos_melhoria,
        "otimos_locais_distintos": otimos,
        "quantidade_otimos_locais_distintos": len(otimos),
        "execucoes_com_otimo_local_completo": execucoes,
        "historico": list(problema.historico),
    }


def validar_vizinhancas(problema):
    """
    Valida os tres operadores e devolve seus tamanhos de vizinhanca.
    """
    vizinhanca_original = problema.vizinhanca
    solucao = list(range(1, problema.n + 1))
    esperado_ativos = set(solucao)
    resultado = {}

    try:
        for nome in VIZINHANCAS:
            problema.definir_vizinhanca(nome)
            vizinhos = [
                tuple(problema.aplicar(solucao, mov))
                for mov in problema.movimentos()
            ]

            if any(len(v) != problema.n or set(v) != esperado_ativos for v in vizinhos):
                raise AssertionError(f"A vizinhanca {nome!r} gerou uma solucao invalida")
            if len(vizinhos) != len(set(vizinhos)):
                raise AssertionError(f"A vizinhanca {nome!r} gerou vizinhos duplicados")

            if nome in ("swap", "2opt"):
                esperado = problema.n * (problema.n - 1) // 2
            else:
                esperado = (problema.n - 1) ** 2

            if len(vizinhos) != esperado:
                raise AssertionError(
                    f"Vizinhanca {nome!r}: esperava {esperado} movimentos, "
                    f"mas encontrei {len(vizinhos)}"
                )

            resultado[nome] = {
                "tamanho": len(vizinhos),
                "formula": "n(n-1)/2" if nome in ("swap", "2opt") else "(n-1)^2",
                "permutacoes_validas": True,
                "sem_duplicatas": True,
            }
    finally:
        problema.definir_vizinhanca(vizinhanca_original)

    return resultado

