
import math
import random

VIZINHANCAS = ("swap", "insertion", "2opt")


def arredondar(x):
    """Arredonda para o inteiro mais proximo, com .5 indo para cima.
    Nao usamos round() do Python porque ele arredonda .5 para o numero PAR:
    """
    return math.floor(x + 0.5)


def _numero(texto):
    """Converte o texto do arquivo em numero: int se for inteiro, float se nao."""
    valor = float(texto)
    return int(valor) if valor.is_integer() else valor


class OrcamentoEsgotado(Exception):
    """Sinal de parada: o orcamento de avaliacoes acabou.

    Lancado pelo proprio problema quando um algoritmo tenta avaliar uma
    solucao alem do orcamento. O algoritmo so precisa captura-lo e encerrar.
    """


class ProblemaManutencao:

    def __init__(self, caminho_arquivo, vizinhanca="swap", incremental=True):
        # --- 1. Ler o arquivo inteiro e quebrar em "pedacos" ---
        with open(caminho_arquivo) as arquivo:
            tokens = arquivo.read().split()

        self.n = int(tokens[0])
        valores = [_numero(tok) for tok in tokens[1:]]

        esperado = 2 + 5 * self.n
        if len(valores) != esperado:
            raise ValueError(
                f"Arquivo {caminho_arquivo}: esperava {esperado} numeros "
                f"apos o n, encontrei {len(valores)}."
            )

        # --- 2. Guardar os dados em listas indexadas pelo numero do ativo ---
        # Posicao 0 = base (p0 = 0; prazo e peso nao se aplicam).
        self.x = [valores[0]]
        self.y = [valores[1]]
        self.p = [0]
        self.d = [0]
        self.w = [0]

        pos = 2
        for _ in range(self.n):
            xj, yj, pj, dj, wj = valores[pos:pos + 5]
            self.x.append(xj)
            self.y.append(yj)
            self.p.append(pj)
            self.d.append(dj)
            self.w.append(wj)
            pos += 5

        # --- 3. Pre-calcular a matriz de tempos de deslocamento (Equacao 1) ---
        pontos = range(self.n + 1)
        self.t = [
            [arredondar(math.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j]))
             for j in pontos]
            for i in pontos
        ]

        # --- 4. Configuracao da vizinhanca e da avaliacao ---
        self.incremental = incremental
        self.definir_vizinhanca(vizinhanca)

        # --- 5. Estado de uma execucao (zerado por reiniciar) ---
        self.reiniciar(semente=None, orcamento=None)

    # Controle de execucao: contador, orcamento e registro do melhor
    def reiniciar(self, semente, orcamento=None):
        """Prepara uma nova execucao. Chamar no inicio de CADA execucao.

        semente   -> fixa os sorteios (mesma semente = mesma execucao)
        orcamento -> numero maximo de avaliacoes (None = sem limite)
        """
        self.avaliacoes = 0
        self.orcamento = orcamento
        self.rng = random.Random(semente)

        self.melhor_h = math.inf     # melhor valor visto nesta execucao
        self.melhor_s = None         # solucao correspondente
        self.historico = []          # pares (avaliacoes, melhor_h) a cada melhora

        self._cache_s = None         # usado pela avaliacao incremental

    def _contar(self):
        """O UNICO lugar onde o contador de avaliacoes aumenta."""
        if self.orcamento is not None and self.avaliacoes >= self.orcamento:
            raise OrcamentoEsgotado()
        self.avaliacoes += 1

    def _registrar(self, valor, s, mov=None):
        """Guarda a melhor solucao ja vista e a curva de convergencia."""
        if valor < self.melhor_h:
            self.melhor_h = valor
            self.melhor_s = self.aplicar(s, mov) if mov is not None else list(s)
            self.historico.append((self.avaliacoes, valor))

    # Interface usada pelos algoritmos
    def solucao_aleatoria(self):
        """Uma permutacao aleatoria dos ativos 1..n (lista de inteiros)."""
        pi = list(range(1, self.n + 1))
        self.rng.shuffle(pi)
        return pi

    def h(self, pi):
        """Avalia uma solucao completa. CONTA 1 avaliacao."""
        self._contar()
        valor = self.custo(pi)
        self._registrar(valor, pi)
        return valor

    def movimentos(self):
        """Lista de todos os movimentos da vizinhanca, em ordem aleatoria.

        Um movimento e um par de posicoes (i, j). O algoritmo nao precisa
        saber o que ele significa: so o repassa para avaliar_movimento e aplicar.
        """
        movs = list(self._movimentos)
        self.rng.shuffle(movs)
        return movs

    def sucessores(self, s):
        """Percorre todos os sucessores de s, devolvendo (movimento, valor).

        Cada sucessor examinado conta 1 avaliacao. A vizinhanca e percorrida
        em ordem aleatoria (importante para o first improvement).
        """
        for mov in self.movimentos():
            yield mov, self.avaliar_movimento(s, mov)

    def sucessor_aleatorio(self, s):
        """Sorteia UM sucessor de s. Devolve (movimento, valor). Conta 1."""
        mov = self._movimentos[self.rng.randrange(len(self._movimentos))]
        return mov, self.avaliar_movimento(s, mov)

    def aplicar(self, s, mov):
        """Devolve a NOVA solucao obtida aplicando o movimento a s."""
        lo, hi, seg = self._segmento(s, mov)
        return s[:lo] + seg + s[hi + 1:]

    # Operadores do algoritmo genetico

    def cruzamento(self, pai1, pai2):
        """Devolve UM filho
        (lista nova); nao conta avaliacao.

        1. Sorteia um bloco [a, b] e copia esse bloco do pai1 para o filho,
           nas mesmas posicoes.
        2. Preenche as posicoes vazias do filho, da esquerda para a direita,
           com os ativos que ainda nao estao no filho, na ordem em que
           aparecem no pai2.

        Posicoes contando do 0, bloco a=1, b=3):
            pai1  = [1, | 2, 3, 4, | 5]
            pai2  = [5, 4, 2, 1, 3]
            passo 1: filho = [_, 2, 3, 4, _]
            pai2 na ordem, sem os que ja estao (2, 3, 4): 5, 1
            preenche as vagas da esquerda para a direita: posicao 0 -> 5, posicao 4 -> 1
            filho = [5, 2, 3, 4, 1]
        """
        n = self.n
        a = self.rng.randrange(n)
        b = self.rng.randrange(n)
        if a > b:
            a, b = b, a
        return self._ox(pai1, pai2, a, b)

    def _ox(self, pai1, pai2, a, b):
        bloco = pai1[a:b + 1]
        presentes = set(bloco)
        resto = [gene for gene in pai2 if gene not in presentes]   # ordem do pai2
        # vagas a esquerda do bloco, o bloco, e vagas a direita do bloco
        return resto[:a] + bloco + resto[a:]

    def mutacao(self, s):
        """Aplica UM movimento aleatorio da vizinhanca atual. Lista nova; nao conta."""
        mov = self._movimentos[self.rng.randrange(len(self._movimentos))]
        return self.aplicar(s, mov)

    # Vizinhancas
    def definir_vizinhanca(self, nome):
        """Escolhe o operador e pre-gera a lista de movimentos possiveis.

        swap      (i, j), i < j : troca os ativos das posicoes i e j
        insertion (i, j), i != j: retira o ativo da posicao i e o insere na j
        2opt      (i, j), i < j : inverte o trecho entre as posicoes i e j
        """
        if nome not in VIZINHANCAS:
            raise ValueError(f"vizinhanca desconhecida: {nome}")
        self.vizinhanca = nome
        n = self.n
        if nome in ("swap", "2opt"):
            # n(n-1)/2 movimentos
            self._movimentos = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            # n(n-1) pares, mas mover i para i-1 da o mesmo resultado que
            # mover i-1 para i (ambos trocam dois vizinhos). Excluimos um deles:
            # sobram (n-1)^2 movimentos, todos gerando solucoes diferentes.
            self._movimentos = [(i, j) for i in range(n) for j in range(n)
                                if i != j and j != i - 1]

    def tamanho_vizinhanca(self):
        return len(self._movimentos)

    def _segmento(self, s, mov):
        """Descreve o efeito do movimento: so o trecho s[lo..hi] muda.

        Devolve (lo, hi, seg), onde seg e o novo conteudo desse trecho.
        Antes de lo e depois de hi, a solucao continua igual.
        """
        i, j = mov
        if self.vizinhanca == "swap":
            return i, j, [s[j]] + s[i + 1:j] + [s[i]]
        if self.vizinhanca == "2opt":
            return i, j, s[i:j + 1][::-1]
        # insertion
        if i < j:
            return i, j, s[i + 1:j + 1] + [s[i]]
        return j, i, [s[i]] + s[j:i]


    def avaliar_movimento(self, s, mov, limite=math.inf):
        """Valor de h do sucessor aplicar(s, mov), SEM construir o sucessor.

        CONTA 1 avaliacao (e um sucessor examinado).

        limite (opcional): se o valor do sucessor for >= limite, o calculo
        pode parar no meio e devolver um numero >= limite que NAO e o valor
        exato. Isso usa a propriedade do enunciado: o custo acumulado so cresce
        ao longo da sequencia. Quem so quer saber "e melhor que X?" passa
        limite=X. Quem precisa do valor exato (simulated annealing) nao passa.
        """
        self._contar()

        if not self.incremental:
            vizinho = self.aplicar(s, mov)
            valor = self.custo(vizinho)
            self._registrar(valor, vizinho)
            return valor

        C, H = self._prefixos(s)
        lo, hi, seg = self._segmento(s, mov)
        t, p, d, w = self.t, self.p, self.d, self.w
        n = self.n

        # Nunca paramos antes de saber se e um novo melhor global.
        corte = max(limite, self.melhor_h)

        # 1) O comeco s[0..lo-1] nao muda: reaproveita tempo e custo prontos.
        tempo = C[lo]
        total = H[lo]
        anterior = s[lo - 1] if lo > 0 else 0

        # 2) Recalcula o trecho alterado.
        for j in seg:
            tempo += t[anterior][j] + p[j]
            atraso = tempo - d[j]
            if atraso > 0:
                total += w[j] * atraso
                if total >= corte:
                    return total          # poda: ja nao serve
            anterior = j

        # 3) O final s[hi+1..] tem os mesmos ativos, na mesma ordem, so que
        #    todos deslocados no tempo pelo MESMO valor delta. Antes de
        #    recalcular um por um, testamos tres atalhos:
        #    a) delta == 0: o final e identico ao original;
        #    b) todos do final continuam atrasados: cada um paga delta horas a
        #       mais (ou a menos), entao o custo muda delta * (soma dos pesos);
        #    c) todos do final continuam no prazo: custo do final = 0.
        L, W, MinL, MaxL = self._L, self._W, self._MinL, self._MaxL
        for k in range(hi + 1, n):
            j = s[k]
            novo_C = tempo + t[anterior][j] + p[j]
            delta = novo_C - C[k + 1]
            if delta == 0:                                  # atalho (a)
                total += H[n] - H[k]
                break
            if MinL[k] + min(0, delta) >= 0:                # atalho (b)
                total += H[n] - H[k] + delta * W[k]
                break
            if MaxL[k] + max(0, delta) <= 0:                # atalho (c)
                break
            tempo = novo_C
            atraso = tempo - d[j]
            if atraso > 0:
                total += w[j] * atraso
                if total >= corte:
                    return total          # poda
            anterior = j

        self._registrar(total, s, mov)
        return total

    def _prefixos(self, s):
        """Resumos da solucao corrente s, usados pela avaliacao incremental.

        C[k], H[k]  instante e custo acumulado apos os k primeiros ativos
        L[k]        folga negativa do ativo da posicao k: C - d (> 0 = atrasado)
        W[k]        soma dos pesos das posicoes k..n-1
        MinL[k]     menor L entre as posicoes k..n-1
        MaxL[k]     maior L entre as posicoes k..n-1

        Fica guardado (cache) enquanto a solucao corrente for a mesma lista,
        entao e calculado uma vez e reaproveitado por todos os sucessores.
        """
        if s is self._cache_s:
            return self._C, self._H
        t, p, d, w = self.t, self.p, self.d, self.w
        n = self.n
        C = [0] * (n + 1)
        H = [0] * (n + 1)
        L = [0] * n
        tempo = total = anterior = 0
        for k, j in enumerate(s):
            tempo += t[anterior][j] + p[j]
            atraso = tempo - d[j]
            if atraso > 0:
                total += w[j] * atraso
            C[k + 1] = tempo
            H[k + 1] = total
            L[k] = atraso
            anterior = j

        # Somas e minimos/maximos "de tras para frente" (sufixos)
        W = [0] * (n + 1)
        MinL = [math.inf] * (n + 1)
        MaxL = [-math.inf] * (n + 1)
        for k in range(n - 1, -1, -1):
            W[k] = W[k + 1] + w[s[k]]
            MinL[k] = min(MinL[k + 1], L[k])
            MaxL[k] = max(MaxL[k + 1], L[k])

        self._cache_s, self._C, self._H = s, C, H
        self._L, self._W, self._MinL, self._MaxL = L, W, MinL, MaxL
        return C, H


    def custo(self, pi):
        """Calcula h(pi) SEM contar avaliacao. Uso: testes e relatorios.

        Simula o dia da equipe: sai da base, anda ate cada ativo na ordem
        de pi, executa o servico e soma a multa de quem terminou atrasado.
        """
        t, p, d, w = self.t, self.p, self.d, self.w   # atalhos locais (mais rapido)
        tempo = 0        # C: instante em que o ultimo servico terminou
        anterior = 0     # onde a equipe esta (comeca na base)
        total = 0        # h acumulado

        for j in pi:
            tempo += t[anterior][j] + p[j]      # Equacao (2)
            atraso = tempo - d[j]
            if atraso > 0:                      # max(0, Cj - dj)
                total += w[j] * atraso          # Equacao (3)
            anterior = j

        return total

    def cronograma(self, pi):
        """Tabela no formato da Figura 1 do enunciado (para o relatorio, T1)."""
        linhas = [f"{'ativo':>5} {'p':>4} {'d':>5} {'w':>3} {'C':>6} {'w(C-d)+':>8}"]
        tempo, anterior, total = 0, 0, 0
        for j in pi:
            tempo += self.t[anterior][j] + self.p[j]
            multa = self.w[j] * max(0, tempo - self.d[j])
            total += multa
            linhas.append(f"{j:>5} {self.p[j]:>4} {self.d[j]:>5} "
                          f"{self.w[j]:>3} {tempo:>6} {multa:>8}")
            anterior = j
        linhas.append(f"h(pi) = {total}")
        return "\n".join(linhas)