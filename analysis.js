// 学習した辞書（トークンをキーにしたオブジェクト）
let learnedDictionary = {};

// 1. 教師データ(training_data.json)の読み込みと学習
async function trainFromCorpus() {
    try {
        const res = await fetch('word.json');
        if (!res.ok) {
            throw new Error();
        }
        const trainingData = await res.json();

        // 辞書の構築プロセス
        learnedDictionary = {};
        trainingData.forEach(entry => {
            entry.segments.forEach(seg => {
                if (!learnedDictionary[seg.token]) {
                    learnedDictionary[seg.token] = {
                        ...seg,
                        count: 0 // 出現回数をカウント（スコア計算用）
                    };
                }
                learnedDictionary[seg.token].count++;
            });
        });

        // 統計情報の表示
        const vocabSize = Object.keys(learnedDictionary).length;
        document.getElementById('vocabCount').innerText = vocabSize;
        console.log("学習完了: ", learnedDictionary);

    } catch (e) {
        console.error("データの読み込みに失敗しました", e);
    }
}

// 2. 解析アルゴリズム (学習済み辞書を使ったビームサーチ)
function analyzeText(text) {
    // DPテーブル: dp[i] は i文字目までの最適な分割リストを持つ
    const dp = Array(text.length + 1).fill().map(() => []);
    dp[0] = [{ tokens: [], score: 0 }];

    const tokens = Object.values(learnedDictionary);

    for (let i = 0; i < text.length; ++i) {
        if (dp[i].length === 0) continue;

        // A. 学習済み単語とのマッチング
        for (const word of tokens) {
            if (text.startsWith(word.token, i)) {
                const nextIndex = i + word.token.length;

                // TensorFlow.js的な重み付け:
                // 長い単語ほど価値が高く、頻出単語(count)も少し有利にする
                const lengthScore = word.token.length * 10;
                const freqScore = Math.log(word.count + 1);
                const wordScore = lengthScore + freqScore;

                dp[i].forEach(path => {
                    dp[nextIndex].push({
                        tokens: [...path.tokens, { ...word, type: 'known' }],
                        score: path.score + wordScore
                    });
                });
            }
        }

        // B. 未知語（学習データにない文字）の処理
        const char = text[i];
        const nextIndex = i + 1;
        dp[i].forEach(path => {
            dp[nextIndex].push({
                tokens: [...path.tokens, { token: char, pos: "未知語", meaning: "？", type: 'unknown' }],
                score: path.score - 5 // ペナルティ
            });
        });

        // 枝刈り（上位のみ残す）
        if (dp[i + 1].length > 50) {
            dp[i + 1].sort((a, b) => b.score - a.score);
            dp[i + 1] = dp[i + 1].slice(0, 50);
        }
    }

    // 最終結果をソートして返す
    const results = dp[text.length];
    return results.sort((a, b) => b.score - a.score).slice(0, 5);
}

// 3. UIへの描画
function renderResults(candidates) {
    const container = document.getElementById('resultContainer');
    container.innerHTML = '';

    if (candidates.length === 0) {
        container.innerHTML = '<div class="p-4 bg-red-50 text-red-600 rounded">解析できませんでした</div>';
        return;
    }

    // スコアの正規化（バー表示用）
    const maxScore = candidates[0].score;
    const minScore = candidates[candidates.length - 1].score;

    candidates.forEach((cand, idx) => {
        // スコア計算 (0-100%)
        let percent = 100;
        if (candidates.length > 1 && maxScore !== minScore) {
            percent = ((cand.score - minScore) / (maxScore - minScore)) * 100;
        }
        percent = Math.max(percent, 5); // 最低幅確保

        const isTop = idx === 0;

        const html = `
                <div class="bg-white rounded-xl border ${isTop ? 'border-blue-500 ring-2 ring-blue-100 shadow-lg' : 'border-gray-200'} overflow-hidden transition hover:shadow-md">
                    <div class="px-4 py-3 bg-gray-50 border-b border-gray-100 flex justify-between items-center">
                        <div class="flex items-center gap-2">
                            <span class="text-xs font-bold px-2 py-1 rounded text-white ${isTop ? 'bg-blue-600' : 'bg-gray-400'}">候補 ${idx + 1}</span>
                            ${isTop ? '<span class="text-xs text-blue-600 font-bold">★ 推奨</span>' : ''}
                        </div>
                        <div class="flex items-center gap-2 w-32">
                            <div class="h-1.5 flex-1 bg-gray-200 rounded-full overflow-hidden">
                                <div class="h-full bg-blue-500" style="width: ${percent}%"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="p-4">
                        <div class="flex flex-wrap gap-2 mb-4">
                            ${cand.tokens.map(t => `
                                <div class="flex flex-col items-center">
                                    <span class="${t.type === 'unknown' ? 'bg-gray-100 text-gray-400 border-dashed' : 'bg-blue-50 text-blue-700'} border px-3 py-1 rounded-lg font-bold text-lg">
                                        ${t.token}
                                    </span>
                                    <span class="text-[10px] text-gray-400 mt-1">${t.pos}</span>
                                </div>
                            `).join('')}
                        </div>

                        <div class="border rounded-lg overflow-hidden text-sm">
                            <table class="w-full text-left">
                                <thead class="bg-gray-100 text-gray-500 text-xs uppercase">
                                    <tr>
                                        <th class="px-3 py-2">形態素</th>
                                        <th class="px-3 py-2">品詞</th>
                                        <th class="px-3 py-2">意味・情報</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-gray-100">
                                    ${cand.tokens.map(t => `
                                        <tr class="hover:bg-gray-50">
                                            <td class="px-3 py-2 font-bold ${t.type === 'unknown' ? 'text-gray-400' : 'text-gray-800'}">${t.token}</td>
                                            <td class="px-3 py-2 text-gray-600">${t.pos}</td>
                                            <td class="px-3 py-2 text-gray-500">${t.meaning}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                `;
        container.innerHTML += html;
    });
}

document.getElementById('analyzeBtn').addEventListener('click', () => {
    const text = document.getElementById('inputText').value.trim();
    if (!text) return;

    const results = analyzeText(text);
    renderResults(results);
});

// 起動時の学習実行
trainFromCorpus();