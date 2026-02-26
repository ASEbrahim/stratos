# StratOS Thumbs Up/Down Feedback + Training Fix

## Overview
6 files to patch. Apply changes in order.

---

## PATCH 1: train_lora.py — Fix meta device crash (CRITICAL)

The `device_map={"": 0}` fix is already on line 254, but after `get_peft_model()` wraps the model, some params end up on meta device. The Trainer then sees `hf_device_map` and refuses to move them.

**Find this block (~line 295):**
```python
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
```

**Replace with:**
```python
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Force all parameters onto GPU 0 after PEFT wrapping
    # PEFT can leave some params on meta device, and the Trainer skips
    # device placement when it sees hf_device_map — causing gradient crashes
    if is_rocm or True:  # Always do this to be safe
        if hasattr(model, 'hf_device_map'):
            del model.hf_device_map
        model = model.to("cuda:0")
        logger.info("Forced all parameters to cuda:0 (meta device cleanup)")
    
    # Load data
```

---

## PATCH 2: main.py — Accept thumbs_up/thumbs_down actions

**Find this line (~line 1738):**
```python
                        if action not in ("click", "dismiss", "rate", "save"):
```

**Replace with:**
```python
                        if action not in ("click", "dismiss", "rate", "save", "thumbs_up", "thumbs_down"):
```

---

## PATCH 3: database.py — Include thumbs in feedback-for-scoring queries

**Find the positive signals query (~line 588):**
```python
        # Positive signals: saved items + highly rated items + clicks on high-scored items
        cursor.execute("""
            SELECT DISTINCT title, ai_score, user_score, category, root, action
            FROM user_feedback
            WHERE created_at > ?
              AND (
                  action = 'save'
                  OR (action = 'rate' AND user_score >= 7.0)
                  OR (action = 'click' AND ai_score >= 6.0)
              )
            ORDER BY created_at DESC
            LIMIT ?
        """, (since, limit))
        positive = [dict(row) for row in cursor.fetchall()]
```

**Replace with:**
```python
        # Positive signals: saved items + highly rated items + clicks on high-scored + thumbs up
        cursor.execute("""
            SELECT DISTINCT title, ai_score, user_score, category, root, action
            FROM user_feedback
            WHERE created_at > ?
              AND (
                  action = 'save'
                  OR action = 'thumbs_up'
                  OR (action = 'rate' AND user_score >= 7.0)
                  OR (action = 'click' AND ai_score >= 6.0)
              )
            ORDER BY created_at DESC
            LIMIT ?
        """, (since, limit))
        positive = [dict(row) for row in cursor.fetchall()]
```

**Find the negative signals query (~line 604):**
```python
        # Negative signals: dismissed items + low-rated items
        cursor.execute("""
            SELECT DISTINCT title, ai_score, user_score, category, root, action
            FROM user_feedback
            WHERE created_at > ?
              AND (
                  action = 'dismiss'
                  OR (action = 'rate' AND user_score IS NOT NULL AND user_score <= 4.0)
              )
            ORDER BY created_at DESC
            LIMIT ?
        """, (since, limit))
        negative = [dict(row) for row in cursor.fetchall()]
```

**Replace with:**
```python
        # Negative signals: dismissed items + low-rated items + thumbs down
        cursor.execute("""
            SELECT DISTINCT title, ai_score, user_score, category, root, action
            FROM user_feedback
            WHERE created_at > ?
              AND (
                  action = 'dismiss'
                  OR action = 'thumbs_down'
                  OR (action = 'rate' AND user_score IS NOT NULL AND user_score <= 4.0)
              )
            ORDER BY created_at DESC
            LIMIT ?
        """, (since, limit))
        negative = [dict(row) for row in cursor.fetchall()]
```

---

## PATCH 4: export_training.py — Export thumbs as high-confidence training data

**Find this block (~line 98):**
```python
    # 2. Strong implicit signals (saves = score ≥ 7.0, dismissals = score ≤ 2.0)
    cursor.execute("""
        SELECT f.news_id, f.title, f.url, f.root, f.category,
               f.ai_score, f.action, f.created_at,
               n.summary, n.source, n.score
        FROM user_feedback f
        LEFT JOIN news_items n ON f.news_id = n.id
        WHERE f.action IN ('save', 'dismiss')
        ORDER BY f.created_at DESC
    """)
```

**Replace with:**
```python
    # 2. Strong implicit signals (saves, dismissals, thumbs up/down)
    cursor.execute("""
        SELECT f.news_id, f.title, f.url, f.root, f.category,
               f.ai_score, f.user_score, f.action, f.created_at,
               n.summary, n.source, n.score
        FROM user_feedback f
        LEFT JOIN news_items n ON f.news_id = n.id
        WHERE f.action IN ('save', 'dismiss', 'thumbs_up', 'thumbs_down')
        ORDER BY f.created_at DESC
    """)
```

**Then find (in the same for-loop, ~line 114):**
```python
        if action == 'save' and original_score < 6.0:
```

**Add BEFORE that line (so the thumbs checks come first):**
```python
        if action == 'thumbs_up':
            # Thumbs up = user explicitly approves, target 9.0
            target = row.get('user_score') or 9.0
            if abs(target - original_score) >= 1.5:
                training_items.append({
                    'title': row.get('title', ''),
                    'summary': row.get('summary', '') or '',
                    'source': row.get('source', '') or '',
                    'category': row.get('category', '') or '',
                    'root': row.get('root', '') or '',
                    'local_score': original_score,
                    'target_score': target,
                    'reason': 'User gave thumbs up — this content is relevant and valuable',
                    'delta': abs(target - original_score),
                    'signal_type': 'thumbs_up',
                })
            continue
        elif action == 'thumbs_down':
            # Thumbs down = user explicitly rejects, target 1.0
            target = row.get('user_score') or 1.0
            if abs(target - original_score) >= 1.5:
                training_items.append({
                    'title': row.get('title', ''),
                    'summary': row.get('summary', '') or '',
                    'source': row.get('source', '') or '',
                    'category': row.get('category', '') or '',
                    'root': row.get('root', '') or '',
                    'local_score': original_score,
                    'target_score': target,
                    'reason': 'User gave thumbs down — this content is noise/irrelevant',
                    'delta': abs(target - original_score),
                    'signal_type': 'thumbs_down',
                })
            continue
        
```

---

## PATCH 5: app.js — Add thumbsUp/thumbsDown functions

**Find this block (~line 320):**
```javascript
function dismissSignal(idx) {
    const item = (window.currentItems || [])[idx];
    if (!item) return;
    item._dismissed = true;
    _sendFeedback(item, 'dismiss');
    if (typeof showToast === 'function') showToast('Dismissed — scorer will learn from this', 'info', 2000);
    renderFeed();
}
```

**Add AFTER it:**
```javascript

function thumbsUpSignal(idx) {
    const item = (window.currentItems || [])[idx];
    if (!item) return;
    item._thumbs = 'up';
    _sendFeedback(item, 'thumbs_up', 9.0);
    if (typeof showToast === 'function') showToast('👍 Marked as relevant — will train scorer', 'success', 2000);
    renderFeed();
}

function thumbsDownSignal(idx) {
    const item = (window.currentItems || [])[idx];
    if (!item) return;
    item._thumbs = 'down';
    _sendFeedback(item, 'thumbs_down', 1.0);
    if (typeof showToast === 'function') showToast('👎 Marked as noise — will train scorer', 'info', 2000);
    renderFeed();
}
```

---

## PATCH 6: feed.js — Add thumbs buttons to news card template

**Find this block in the card template (~line 604):**
```javascript
            <div class="flex items-center gap-3 mt-2">
                <button onclick="toggleSaveSignal(${idx})" class="text-[10px] ${isSignalSaved(item) ? 'text-emerald-400' : 'text-slate-500 hover:text-emerald-400'} flex items-center gap-1 transition-colors">
                    <i data-lucide="${isSignalSaved(item) ? 'bookmark-check' : 'bookmark'}" class="w-3 h-3"></i> ${isSignalSaved(item) ? 'Saved' : 'Save'}
                </button>
```

**Replace with:**
```javascript
            <div class="flex items-center gap-3 mt-2">
                <button onclick="thumbsUpSignal(${idx})" class="text-[10px] ${item._thumbs === 'up' ? 'text-emerald-400 font-bold' : 'text-slate-500 hover:text-emerald-400'} flex items-center gap-1 transition-colors" title="Relevant — train scorer to score higher">
                    <i data-lucide="thumbs-up" class="w-3 h-3"></i>
                </button>
                <button onclick="thumbsDownSignal(${idx})" class="text-[10px] ${item._thumbs === 'down' ? 'text-red-400 font-bold' : 'text-slate-500 hover:text-red-400'} flex items-center gap-1 transition-colors" title="Noise — train scorer to score lower">
                    <i data-lucide="thumbs-down" class="w-3 h-3"></i>
                </button>
                <span class="text-slate-700">|</span>
                <button onclick="toggleSaveSignal(${idx})" class="text-[10px] ${isSignalSaved(item) ? 'text-emerald-400' : 'text-slate-500 hover:text-emerald-400'} flex items-center gap-1 transition-colors">
                    <i data-lucide="${isSignalSaved(item) ? 'bookmark-check' : 'bookmark'}" class="w-3 h-3"></i> ${isSignalSaved(item) ? 'Saved' : 'Save'}
                </button>
```

---

## How It All Flows

```
User clicks 👍 on news card
  → thumbsUpSignal(idx)
    → _sendFeedback(item, 'thumbs_up', 9.0)
      → POST /api/feedback {action: "thumbs_up", user_score: 9.0, ...}
        → db.save_feedback() → user_feedback table

Next scan cycle:
  → db.get_feedback_for_scoring() pulls thumbs_up as positive signal
  → Injected into scorer prompt → immediate scoring improvement

Next training cycle:
  → export_training.py includes thumbs as high-confidence examples
  → train_lora.py fine-tunes model (with fixed device_map!)
  → Permanent model improvement
```

## Training Fix Summary

The crash `RuntimeError: Function MmBackward0 returned an invalid gradient at index 1 - expected device meta but got cuda:0` persists because:

1. ✅ `device_map={"": 0}` loads model onto GPU correctly
2. ❌ `get_peft_model()` wrapping leaves some params on meta
3. ❌ Trainer sees `hf_device_map` attribute → skips moving params → crash

Fix: After PEFT wrapping, delete `hf_device_map` and force `.to("cuda:0")`.
