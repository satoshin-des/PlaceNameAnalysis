let dictionaryData = [];

    // word.json の読み込み
    async function loadDictionary() {
        try {
            const response = await fetch('word.json');
            if (!response.ok) throw new Error('word.json が見つかりません。');
            const rawData = await response.json();
            
            // { "や-くも": "たくさん-雲" } 形式を [{word: "や-くも", mean: "たくさん-雲"}] に変換
            dictionaryData = Object.entries(rawData).map(([word, mean]) => ({
                word: word,
                mean: mean
            }));

            document.getElementById('status').innerText = `辞書ロード完了: ${dictionaryData.length} 件の定義`;
            analyze();
        } catch (error) {
            document.getElementById('status').innerText = 'エラー: ' + error.message;
            console.error(error);
        }
    }

    // TensorFlow.js を使用した文字列の類似度計算
    async function calculateSimilarity(text1, text2) {
        // ハイフンを除去して比較用の文字列を作成
        const clean1 = text1.replace(/-/g, '');
        const clean2 = text2.replace(/-/g, '');
        
        const combined = clean1 + clean2;
        if (!combined) return 0;

        const chars = Array.from(new Set(combined.split('')));
        const vectorize = (str) => chars.map(c => str.includes(c) ? 1 : 0);
        
        return tf.tidy(() => {
            const v1 = tf.tensor1d(vectorize(clean1));
            const v2 = tf.tensor1d(vectorize(clean2));
            
            const dot = v1.dot(v2);
            const norm1 = v1.norm();
            const norm2 = v2.norm();
            const cosineSim = dot.div(norm1.mul(norm2));
            
            const score = cosineSim.dataSync()[0];
            return isNaN(score) ? 0 : score;
        });
    }

    async function analyze() {
        const input = document.getElementById('targetInput').value.trim();
        const resultsDiv = document.getElementById('results');
        const status = document.getElementById('status');

        if (!input) {
            resultsDiv.innerHTML = '<div class="empty-state">入力待ち...</div>';
            return;
        }

        // Intl.Segmenter で入力文を分かち書き
        const segmenter = new Intl.Segmenter('ja-JP', { granularity: 'word' });
        const tokens = Array.from(segmenter.segment(input)).map(t => t.segment);

        let scoredItems = [];

        // 辞書データと照合（推論）
        for (const item of dictionaryData) {
            let maxScore = 0;
            const cleanDictWord = item.word.replace(/-/g, '');

            // 入力された文の中の各単語と、辞書の単語を比較
            for (const token of tokens) {
                // 完全一致ならスコア1.0
                if (token === cleanDictWord) {
                    maxScore = 1.0;
                    break;
                }
                // TensorFlow.jsによる類似度計算
                const sim = await calculateSimilarity(token, cleanDictWord);
                if (sim > maxScore) maxScore = sim;
            }

            scoredItems.push({ 
                ...item, 
                score: maxScore 
            });
        }

        // スコア順にソートして上位10件を表示
        scoredItems.sort((a, b) => b.score - a.score);
        const top10 = scoredItems.slice(0, 10).filter(i => i.score > 0.1);

        // UI表示の更新
        resultsDiv.innerHTML = '';
        if (top10.length === 0) {
            resultsDiv.innerHTML = '<div class="empty-state">該当する形態素が見つかりません</div>';
        } else {
            top10.forEach(item => {
                const itemEl = document.createElement('div');
                itemEl.className = 'result-item';
                itemEl.innerHTML = `
                    <div class="morpheme-info">
                        <div class="morpheme-header">
                            <span class="word">${item.word}</span>
                            <span class="score-badge">${Math.round(item.score * 100)}% Match</span>
                        </div>
                        <span class="meaning">${item.mean}</span>
                    </div>
                `;
                resultsDiv.appendChild(itemEl);
            });
        }
        status.innerText = `推論完了`;
    }

    // イベント設定
    document.getElementById('targetInput').addEventListener('input', analyze);

    // 起動時に辞書をロード
    window.onload = loadDictionary;
