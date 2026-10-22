from problema import ProblemaManutencao
from busca_local import greedy_descent

arquivo = "manutencao_pequena.txt"
orcamento = 50_000

for vizinhanca in ("swap", "insertion", "2opt"):
    for estrategia in ("best", "first"):
        problema = ProblemaManutencao(
            arquivo,
            vizinhanca=vizinhanca,
            incremental=True
        )

        resultado = greedy_descent(
            problema,
            estrategia=estrategia,
            semente=42,
            orcamento=orcamento
        )

        print(
            vizinhanca,
            estrategia,
            "melhor h =", resultado["melhor_h"],
            "avaliações =", resultado["avaliacoes"],
            "passos =", resultado["passos_melhoria"]
        )