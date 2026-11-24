# 第6回 解答

## 演習

<!-- [tool call] の行は、表示された行をそのまま貼り付けてください。終わらなかった STEP は、空欄のままでかまいません。 -->

- STEP 1 のツールの名前：`glossary_search_glossary`
  - `uv run python -m app.agent --list-tools`（第5回までのエージェント、`search_materials`）と `uv run python -m app.mcp_agent --list-tools`（MCP 経由、`glossary_search_glossary`）を比較。どちらも名前・説明文・引数の同じ形でモデルに渡ることを確認した。
- STEP 2 の [tool call]：`--trace`（実 LLM）は未実施（ローカルに `GEMINI_API_KEY` なし）。同等の検証として FunctionModel による機械的呼び出しで `glossary_search_glossary` 相当の配線を確認（宿題の欄に記載）
- STEP 3 の [tool call]：未実施（同上。`mcp_servers.json` への `fetch` 追加は設定内容のみ確認）
- STEP 4 の [tool call]：未実施（同上。`url` 接続型の `aws` サーバーの設定書式のみ確認）
- 気づき・疑問：`--list-tools` は API キーなしで動き、サーバーが起動して返すツールの一覧（名前・docstring・引数のスキーマ）をそのまま確認できる。MCP のツール名が「サーバー名_ツール名」で命名されるため、複数サーバーで同名ツールがあっても衝突しない設計になっている。

## 宿題

<!-- 基礎・実装・発展のどれか1つに取り組めば完了です。 -->

- 取り組んだレベル：基礎
- やったこと：`app/mcp_server.py` の TODO の下に `server.tool()(tools.get_lecture_summary)` の 1 行を書き足した。エージェント側（`app/mcp_servers.json`、`app/mcp_agent.py`）は変更していない。変更後、`git status` で変更ファイルが `app/mcp_server.py` だけであることを確認した。
- 結果：
  - `uv run python -m app.mcp_agent --list-tools` の出力：

    ```text
    [server] glossary: コマンドで起動（uv run python -m app.mcp_server）
    モデルに見えているツール: 2 個

    [glossary_search_glossary]
      講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。
      引数: {"query": {"title": "Query", "type": "string"}}

    [glossary_get_lecture_summary]
      指定した回（1始まりの整数）の講義の要点を返す。
      引数: {"lecture": {"title": "Lecture", "type": "integer"}}
    ```

  - エージェント側を変えなくても、起動時にサーバーへツール一覧を問い合わせるため、追加した `glossary_get_lecture_summary` が「2 個」の 2 つ目として一覧に現れた。
  - `uv run pytest -q` → `21 passed`（すべて通過）。
  - `--trace` のモデル応答は未検証（ローカルに `GEMINI_API_KEY` がないため）だが、pydantic-ai の `FunctionModel`（テスト用の疑似モデル）で「モデルが `glossary_get_lecture_summary` を呼ぶ」ステップを機械的に再現し、MCP 経由の呼び出しと結果返却の配線を確認した：

    ```text
    --- trace ---
    [tool call]   glossary_get_lecture_summary({'lecture': 2})
    [tool return] glossary_get_lecture_summary -> # 第2回 ファインチューニングとLoRA
                  ## この回のねらい
                  - 事前学習とファインチューニングの違い、ファインチューニングが向く場面と向かない場面を理解する
                  - LoRAの考え方を理解し、Google Colab上で小さなモデルを…
    [usage] 入力 160 tokens / 出力 9 tokens / モデル応答 2 回
    ```

  - つまり、エージェント側を 1 行も変えずに、追加したツールがサーバー経由で呼べ、第2回の要点が返っている。未検証なのは「実際の LLM がこのツールを選ぶか」の部分のみ。
- 気づき・疑問：サーバー側で `server.tool()` を付けて公開するだけでクライアント側の変更が不要なのは、MCP がツール一覧を起動時に動的に取得するプロトコルだから。docstring と型ヒント（`lecture: int`）がそのまま説明文と引数スキーマとしてモデルに渡るため、ツールの説明の書き方がそのまま呼び出し精度に影響すると感じた。
