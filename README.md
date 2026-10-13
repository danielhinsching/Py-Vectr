# Vetorizador

Converte imagens (JPG, PNG, WEBP, GIF, BMP, TIFF, HEIC, AVIF...) em vetor, localmente no seu PC.

## Como usar

Dê dois cliques em **`iniciar.bat`**. Na primeira vez ele instala as dependências; depois abre o navegador em http://127.0.0.1:5000.

Requer Python instalado. O `iniciar.bat` funciona em qualquer pasta, então pode clonar o repositório ou baixar o ZIP (pasta `Vetorizador-main`) sem renomear nada.

Arraste uma imagem para a janela (ou clique na área à esquerda, ou cole com Ctrl+V). O resultado atualiza sozinho a cada ajuste.

## Modos

### SVG Padrão
Para design, impressão e web. Mantém as cores.

| Tipo | Quando usar |
|---|---|
| **Logo** | Poucas cores chapadas. Detecta automaticamente as cores reais (até o limite escolhido) e elimina os halos de antialiasing/JPG. |
| **Ilustração** | Artes com mais cores. |
| **Foto** | Mantém todas as cores (arquivo maior). |
| **P&B** | Traço preto e branco de alta precisão (Potrace). |

"Remover fundo" deixa transparente o fundo conectado à borda da imagem.

### Cortadora a Laser
Gera **SVG em milímetros** e **DXF** com contornos fechados.

- **Corte (linha)**: linhas vermelhas finas (hairline 0,001"), sem preenchimento.
- **Gravação (preench.)**: formas preenchidas em preto.
- **Tamanho real**: defina a largura em mm; a altura segue a proporção. Se a imagem tiver DPI salvo, ele é usado por padrão.
- **O que traçar**: *Áreas escuras* (limiar claro/escuro) ou *Tudo exceto o fundo* (bom para logos coloridos).
- **Somente contorno externo**: corta só a silhueta (adesivos, chaveiros).
- A barra superior mostra o comprimento total de corte.

SVG: LightBurn, Glowforge, xTool, Inkscape. DXF: RDWorks, LaserCAD, AutoCAD.

### Arrumar bordas (os dois modos)
**Endireitar bordas** deixa as bordas retas, com cantos vivos na interseção das retas vizinhas.

- **Só retas**: cada borda vira uma única reta de canto a canto, mesmo que o original seja ondulado. Curvas também viram retas.
- **Preservar curvas**: endireita só o que já é quase reto; círculos e arcos continuam curvos.
- **Tolerância**: tamanho das ondas a eliminar, em pixels da imagem original. Aumente até as ondas sumirem.
- **Alinhar retas quase horizontais/verticais**: retas a menos de 3° do eixo ficam exatamente retas.

## Arquivos

- `vectorizer.py`: motor de vetorização (vtracer para cor, Potrace para P&B/laser)
- `app.py`: servidor local (Flask)
- `static/index.html`: interface
