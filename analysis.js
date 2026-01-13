let dictionary = [];

// 1. 辞書のロード
async function loadDictionary() {
    try {
        const res = await fetch('word.json');
        if (!res.ok) {
            throw new Error();
        }

        dictionary = await res.json();
    } catch (e) {
        console.log("word.jsonが存在しません");
    }
}

// 2. 探索アルゴリズム (Beam Search風)
function findSegmentations(text) {
    const dp = Array(text.length + 1).fill().map(() => []);
    dp[0] = [{ tokens: [], score: 0 }];

    for (let i = 0; i < text.length; ++i) {
        if (dp[i].length === 0) {
            continue;
        }

        // A. 辞書マッチング
        for (const word of dictionary) {
            if (text.startsWith(word.token, i)) {
                const nextIndex = i + word.token.length;
                dp[i].forEach(path => {
                    dp[nextIndex].push({
                        tokens: [...path.tokens, { ...word, type: 'known' }],
                        score: path.score + (word.token.length ** 2) * 10
                    });
                });
            }
        }

        // B. 未知語処理（1文字飛ばし）
        const char = text[i];
        const nextIndex = i + 1;
        dp[i].forEach(path => {
            dp[nextIndex].push({
                tokens: [...path.tokens, { token: char, meaning: "（辞書なし）", example: "-", type: 'unknown' }],
                score: path.score - 5
            });
        });

        // 上位20件に絞る（枝刈り）
        if (dp[i + 1].length > 20) {
            dp[i + 1].sort((a, b) => b.score - a.score);
            dp[i + 1] = dp[i + 1].slice(0, 20);
        }
    }
    return dp[text.length];
}

// 3. ランク付け (TF.js使用)
function rankResults(candidates) {
    return tf.tidy(() => {
        if (candidates.length === 0) {
            return [];
        }

        const scores = tf.tensor1d(candidates.map(c => c.score));
        const min = scores.min();
        const max = scores.max();
        const range = max.sub(min);

        const normalized = range.dataSync()[0] === 0
            ? scores.sub(min).add(1)
            : scores.sub(min).div(range);

        const normalizedScores = normalized.dataSync();

        const ranked = candidates.map((c, i) => ({
            ...c,
            confidence: normalizedScores[i]
        })).sort((a, b) => b.score - a.score);

        return ranked.slice(0, 10);
    });
}

// 4. UI描画
function renderResults(results) {
    const container = document.getElementById('resultContainer');
    container.innerHTML = '';
    container.classList.remove('hidden');

    if (results.length === 0) {
        container.innerHTML = '<div class="text-center text-gray-500 py-10">解析候補が見つかりませんでした。</div>';
        return;
    }

    results.forEach((res, index) => {
        const confidencePercent = Math.round(res.confidence * 100);
        const isTop = index === 0;

        // カード生成
        const card = document.createElement('div');
        // 枠線の強調ロジックは残しつつ、中身は全て表示する
        card.className = `bg-white rounded-xl border-2 ${isTop ? 'border-blue-500 ring-4 ring-blue-50/50' : 'border-gray-200'} p-6 transition`;

        // ヘッダー
        let html = `
                    <div class="flex items-center justify-between mb-4 border-b border-gray-100 pb-3">
                        <div class="flex items-center gap-3">
                            <span class="text-sm font-bold ${isTop ? 'bg-blue-600 text-white' : 'bg-gray-500 text-white'} px-3 py-1 rounded-full">
                                候補 ${index + 1}
                            </span>
                            ${isTop ? '<span class="text-xs font-bold text-blue-600">★ 最有力</span>' : ''}
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-xs text-gray-400">類似度</span>
                            <div class="w-24 bg-gray-100 rounded-full h-2">
                                <div class="bg-blue-500 h-2 rounded-full" style="width: ${Math.max(confidencePercent, 5)}%"></div>
                            </div>
                            <span class="text-xs font-mono text-gray-500 w-8 text-right">${confidencePercent}%</span>
                        </div>
                    </div>
                `;

        // トークン（青いタグ）エリア
        html += `<div class="flex flex-wrap items-center gap-2 mb-5">`;
        res.tokens.forEach(t => {
            const isUnknown = t.type === 'unknown';
            html += `
                        <span class="${isUnknown ? 'unknown-token' : 'bg-blue-50 border border-blue-200 text-blue-700'} px-3 py-1.5 rounded-lg font-bold text-lg">
                            ${t.token}
                        </span>
                    `;
        });
        html += `</div>`;

        // 詳細リスト（ここを全候補で表示に変更）
        // 読みやすさのためにテーブルライクなレイアウトに変更
        html += `
                    <div class="bg-gray-50 rounded-lg overflow-hidden border border-gray-100">
                        <div class="px-4 py-2 bg-gray-100 text-xs font-bold text-gray-500 uppercase tracking-wider flex">
                            <span class="w-24">単語</span>
                            <span class="flex-1">意味・用例</span>
                        </div>
                `;

        res.tokens.filter(t => t.type !== 'unknown').forEach(t => {
            html += `
                        <div class="details-row px-4 py-3 flex gap-4 border-t border-gray-100 items-start">
                            <span class="font-bold text-gray-800 w-24 pt-0.5">${t.token}</span>
                            <div class="flex-1">
                                <p class="text-sm text-gray-700 mb-1 font-medium">${t.meaning}</p>
                                <p class="text-xs text-gray-500 italic">Example: ${t.example}</p>
                            </div>
                        </div>
                    `;
        });

        html += `</div>`; // 閉じる

        card.innerHTML = html;
        container.appendChild(card);
    });
}

// イベントリスナー
document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const text = document.getElementById('inputText').value.trim();
    if (!text) return;
    const candidates = findSegmentations(text);
    const ranked = rankResults(candidates);
    renderResults(ranked);
});

loadDictionary();