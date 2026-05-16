/* ============================================================================
 * NeuralForge — schema-driven Builder form renderer
 * ============================================================================
 *
 * One module, one job: given the `{groups, defs}` descriptor that
 * /api/configs/schema returns and a current config object, render HTML
 * inputs that read/write into the config and emit `oninput` events
 * the Builder picks up to refresh its YAML preview + save handler.
 *
 * The renderer is intentionally dumb. All policy (which group is
 * visible for which model_type, which fields are required, what their
 * bounds are, etc.) lives in the descriptor — derived from Pydantic
 * once on the server. Adding a new model family becomes a Pydantic-only
 * change.
 *
 * Field kinds (matched against schema introspector output):
 *
 *   int / int?         — <input type="number" step="1">
 *   number / number?   — <input type="number" step="any">
 *   string / string?   — <input type="text">
 *   bool               — <input type="checkbox">
 *   enum / enum?       — <select>
 *   list[primitive]    — repeatable inputs + add/remove buttons
 *   list[object]       — repeatable subform via defs lookup
 *   list[any]          — opaque <textarea> with JSON parse on commit
 *   object             — nested subform
 *   any                — opaque JSON textarea
 *
 * Cross-field hooks (HFPipeline 4-bit/8-bit mutual exclusion, dataset
 * Browse button) get a registration API at the bottom — they fire after
 * the renderer has built the form so they can hook into specific paths.
 * ============================================================================ */

const NF_FORM = (function () {
  /* ----- public API --------------------------------------------------- */

  /**
   * Render the entire form into `hostEl` from a `{groups, defs}` descriptor
   * and a current config object. `onChange(path, value)` fires for every
   * input change so the caller can update its live YAML preview.
   *
   * Returns a small handle exposing:
   *   getConfig() — current config (after all edits)
   *   refresh()   — re-evaluate visible_when and re-render hidden groups
   *   destroy()   — clear listeners (idempotent)
   */
  function render({ host, descriptor, initialConfig, onChange, hooks }) {
    if (!host) throw new Error("NF_FORM.render: host element required");
    const state = {
      host,
      descriptor: descriptor || { groups: [], defs: {} },
      config: _deepClone(initialConfig || {}),
      onChange: typeof onChange === "function" ? onChange : () => {},
      hooks: hooks || {},
      groupNodes: new Map(),   // group_id → group DOM container
      listeners: [],
    };

    _renderAllGroups(state);
    _applyVisibility(state);
    _runHooks(state);

    return {
      getConfig: () => _deepClone(state.config),
      refresh:   () => { _applyVisibility(state); },
      destroy:   () => { _teardown(state); },
    };
  }

  return { render };

  /* ----- internals ---------------------------------------------------- */

  /**
   * Build one DOM block per group. Visibility is decided in a separate
   * pass so the same nodes can be hidden/shown without re-rendering when
   * the user flips model.type.
   */
  function _renderAllGroups(state) {
    state.host.innerHTML = "";
    for (const group of state.descriptor.groups) {
      const node = _renderGroup(state, group);
      state.groupNodes.set(group.id, node);
      state.host.appendChild(node);
    }
  }

  function _renderGroup(state, group) {
    const card = document.createElement("div");
    card.className = "card mb-3 nf-form-group";
    card.dataset.groupId = group.id;

    const head = document.createElement("div");
    head.className = "card-head";
    const title = document.createElement("div");
    title.className = "card-title";
    title.textContent = group.title || group.id;
    head.appendChild(title);
    card.appendChild(head);

    const body = document.createElement("div");
    body.className = "nf-form-body";
    body.style.cssText = "display:flex;flex-direction:column;gap:10px";
    for (const f of (group.fields || [])) {
      body.appendChild(_renderField(state, f));
    }
    card.appendChild(body);
    return card;
  }

  /**
   * Render one field record. Dispatches on `kind`. Each control writes
   * to state.config via _setPath() and fires state.onChange().
   */
  function _renderField(state, field) {
    const row = document.createElement("div");
    row.className = "nf-field";
    row.dataset.path = field.path;

    const label = document.createElement("label");
    label.className = "label";
    label.textContent = field.label || field.path.split(".").pop();
    if (field.required) {
      const star = document.createElement("span");
      star.className = "text-faint";
      star.textContent = " *";
      label.appendChild(star);
    }
    row.appendChild(label);

    let control;
    const kind = field.kind || "any";
    if (kind === "bool") {
      control = _checkbox(state, field);
    } else if (kind.startsWith("enum")) {
      control = _select(state, field);
    } else if (kind.startsWith("int") || kind.startsWith("number")) {
      control = _number(state, field);
    } else if (kind.startsWith("string")) {
      control = _text(state, field);
    } else if (kind === "list[primitive]") {
      control = _listPrimitive(state, field);
    } else if (kind === "list[object]") {
      control = _listObject(state, field);
    } else if (kind === "list[any]" || kind === "any" || kind === "object") {
      control = _jsonTextarea(state, field);
    } else {
      control = _text(state, field);
    }
    row.appendChild(control);

    if (field.help) {
      const help = document.createElement("div");
      help.className = "text-xs text-faint mt-1";
      help.textContent = field.help;
      row.appendChild(help);
    }
    return row;
  }

  /* ----- primitive controls ------------------------------------------ */

  function _text(state, field) {
    const inp = document.createElement("input");
    inp.className = "input";
    inp.type = "text";
    const cur = _getPath(state.config, field.path);
    inp.value = cur != null ? String(cur) : (field.default ?? "");
    inp.placeholder = field.default != null ? `(default: ${field.default})` : "";
    inp.addEventListener("input", () => _commit(state, field, inp.value || null));
    return inp;
  }

  function _number(state, field) {
    const inp = document.createElement("input");
    inp.className = "input input-mono";
    inp.type = "number";
    inp.step = field.kind.startsWith("int") ? "1" : "any";
    if (field.min != null) inp.min = field.min;
    if (field.max != null) inp.max = field.max;
    // exclusive_min nudges by 1 step so the browser's native validation
    // matches Pydantic's `gt` semantics.
    if (field.exclusive_min != null) inp.min = field.exclusive_min;
    const cur = _getPath(state.config, field.path);
    if (cur != null) inp.value = cur;
    if (field.default != null) inp.placeholder = `(default: ${field.default})`;
    inp.addEventListener("input", () => {
      const raw = inp.value;
      const v = raw === "" ? null
              : field.kind.startsWith("int") ? parseInt(raw, 10) : parseFloat(raw);
      _commit(state, field, (raw === "" || isNaN(v)) ? null : v);
    });
    return inp;
  }

  function _checkbox(state, field) {
    // Wrap in a flex row so the label sits next to the box rather than above.
    const wrap = document.createElement("label");
    wrap.className = "row-tight";
    wrap.style.cursor = "pointer";
    const inp = document.createElement("input");
    inp.type = "checkbox";
    const cur = _getPath(state.config, field.path);
    inp.checked = (cur != null) ? !!cur : !!field.default;
    inp.addEventListener("change", () => _commit(state, field, !!inp.checked));
    const sub = document.createElement("span");
    sub.className = "text-xs text-faint";
    sub.textContent = field.default != null ? `(default: ${field.default})` : "";
    wrap.appendChild(inp);
    wrap.appendChild(sub);
    return wrap;
  }

  function _select(state, field) {
    const sel = document.createElement("select");
    sel.className = "select";
    const choices = field.choices || [];
    if (field.kind.endsWith("?")) {
      const opt = document.createElement("option");
      opt.value = ""; opt.textContent = "(none)";
      sel.appendChild(opt);
    }
    for (const c of choices) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      sel.appendChild(opt);
    }
    const cur = _getPath(state.config, field.path);
    if (cur != null) sel.value = cur;
    else if (field.default != null) sel.value = field.default;
    sel.addEventListener("change", () => {
      const v = sel.value === "" ? null : sel.value;
      _commit(state, field, v);
      // Selecting a new value may flip visibility for sibling groups
      // (the model.type discriminator drives the 9 arch sub-forms).
      _applyVisibility(state);
    });
    return sel;
  }

  /* ----- list controls ------------------------------------------------ */

  function _listPrimitive(state, field) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:4px";

    const renderRows = () => {
      wrap.innerHTML = "";
      const arr = _getPath(state.config, field.path) || [];
      arr.forEach((val, idx) => {
        const row = document.createElement("div");
        row.className = "row-tight";
        const inp = document.createElement("input");
        inp.className = "input input-mono";
        inp.type = field.item_kind === "int" || field.item_kind === "number"
          ? "number" : "text";
        inp.value = val ?? "";
        inp.addEventListener("input", () => {
          const newArr = (_getPath(state.config, field.path) || []).slice();
          newArr[idx] = field.item_kind === "int" ? parseInt(inp.value, 10)
                      : field.item_kind === "number" ? parseFloat(inp.value)
                      : inp.value;
          _commit(state, field, newArr);
        });
        const rm = document.createElement("button");
        rm.className = "btn btn-ghost btn-xs";
        rm.type = "button";
        rm.textContent = "✕";
        rm.addEventListener("click", () => {
          const newArr = (_getPath(state.config, field.path) || []).slice();
          newArr.splice(idx, 1);
          _commit(state, field, newArr);
          renderRows();
        });
        row.appendChild(inp);
        row.appendChild(rm);
        wrap.appendChild(row);
      });
      const add = document.createElement("button");
      add.className = "btn btn-secondary btn-xs";
      add.type = "button";
      add.textContent = "+ Add";
      add.addEventListener("click", () => {
        const newArr = (_getPath(state.config, field.path) || []).slice();
        newArr.push(field.item_kind === "int" || field.item_kind === "number" ? 0 : "");
        _commit(state, field, newArr);
        renderRows();
      });
      wrap.appendChild(add);
    };
    renderRows();
    return wrap;
  }

  function _listObject(state, field) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:8px";
    const itemDef = state.descriptor.defs[field.item_def] || {};
    const itemProps = itemDef.properties || {};

    const renderRows = () => {
      wrap.innerHTML = "";
      const arr = _getPath(state.config, field.path) || [];
      arr.forEach((item, idx) => {
        const card = document.createElement("div");
        card.style.cssText = "border:1px solid var(--border);border-radius:6px;padding:8px;background:var(--bg-elev)";
        const head = document.createElement("div");
        head.className = "row-tight";
        head.style.justifyContent = "space-between";
        const h = document.createElement("span");
        h.className = "text-xs text-faint";
        h.textContent = `Item ${idx + 1}`;
        const rm = document.createElement("button");
        rm.className = "btn btn-ghost btn-xs";
        rm.type = "button";
        rm.textContent = "✕";
        rm.addEventListener("click", () => {
          const newArr = (_getPath(state.config, field.path) || []).slice();
          newArr.splice(idx, 1);
          _commit(state, field, newArr);
          renderRows();
        });
        head.appendChild(h);
        head.appendChild(rm);
        card.appendChild(head);
        // Render each property as a primitive control bound to this item.
        for (const [pname, pprop] of Object.entries(itemProps)) {
          const subKind = _subKindFromSchemaFragment(pprop);
          const row = document.createElement("div");
          row.className = "row-tight mt-2";
          const lbl = document.createElement("label");
          lbl.className = "text-xs text-muted";
          lbl.style.minWidth = "100px";
          lbl.textContent = pname;
          row.appendChild(lbl);
          const inp = document.createElement("input");
          inp.className = "input input-mono";
          if (subKind === "bool") {
            inp.type = "checkbox";
            inp.checked = !!item[pname];
            inp.addEventListener("change", () => {
              const newArr = (_getPath(state.config, field.path) || []).slice();
              newArr[idx] = { ...newArr[idx], [pname]: inp.checked };
              _commit(state, field, newArr);
            });
          } else if (subKind === "int" || subKind === "number") {
            inp.type = "number";
            inp.step = subKind === "int" ? "1" : "any";
            if (item[pname] != null) inp.value = item[pname];
            if (pprop.default != null) inp.placeholder = String(pprop.default);
            inp.addEventListener("input", () => {
              const newArr = (_getPath(state.config, field.path) || []).slice();
              const raw = inp.value;
              const v = raw === "" ? null
                      : subKind === "int" ? parseInt(raw, 10) : parseFloat(raw);
              newArr[idx] = { ...newArr[idx], [pname]: isNaN(v) ? null : v };
              _commit(state, field, newArr);
            });
          } else {
            inp.type = "text";
            inp.value = item[pname] != null ? String(item[pname]) : "";
            if (pprop.default != null) inp.placeholder = String(pprop.default);
            inp.addEventListener("input", () => {
              const newArr = (_getPath(state.config, field.path) || []).slice();
              newArr[idx] = { ...newArr[idx], [pname]: inp.value };
              _commit(state, field, newArr);
            });
          }
          row.appendChild(inp);
          card.appendChild(row);
        }
        wrap.appendChild(card);
      });
      const add = document.createElement("button");
      add.className = "btn btn-secondary btn-xs";
      add.type = "button";
      add.textContent = "+ Add item";
      add.addEventListener("click", () => {
        const newArr = (_getPath(state.config, field.path) || []).slice();
        // New item starts as an empty object — Pydantic fills in defaults.
        newArr.push({});
        _commit(state, field, newArr);
        renderRows();
      });
      wrap.appendChild(add);
    };
    renderRows();
    return wrap;
  }

  function _subKindFromSchemaFragment(prop) {
    const t = prop.type;
    if (t === "integer") return "int";
    if (t === "number")  return "number";
    if (t === "boolean") return "bool";
    return "string";
  }

  function _jsonTextarea(state, field) {
    // Fallback for object / list[any] / any kinds — opaque JSON with
    // commit on blur so partial edits don't fire parse errors on each
    // keystroke.
    const ta = document.createElement("textarea");
    ta.className = "input input-mono";
    ta.style.minHeight = "80px";
    const cur = _getPath(state.config, field.path);
    if (cur != null) ta.value = JSON.stringify(cur, null, 2);
    else if (field.default != null) ta.placeholder = JSON.stringify(field.default);
    ta.addEventListener("blur", () => {
      if (ta.value.trim() === "") {
        _commit(state, field, null);
        return;
      }
      try { _commit(state, field, JSON.parse(ta.value)); }
      catch (_) { /* leave as-is; save-time validate will surface the error */ }
    });
    return ta;
  }

  /* ----- visibility (discriminated unions) ----------------------------- */

  /**
   * Evaluate each group's `visible_when` rule against the live config
   * and toggle its DOM node. With no rule the group is always visible.
   */
  function _applyVisibility(state) {
    for (const group of state.descriptor.groups) {
      const node = state.groupNodes.get(group.id);
      if (!node) continue;
      const cond = group.visible_when;
      let show = true;
      if (cond && typeof cond === "object") {
        show = Object.entries(cond).every(
          ([p, v]) => String(_getPath(state.config, p)) === String(v),
        );
      }
      node.style.display = show ? "" : "none";
    }
  }

  /* ----- hooks (per-section cross-field behavior) --------------------- */

  function _runHooks(state) {
    if (!state.hooks) return;
    for (const [groupId, fn] of Object.entries(state.hooks)) {
      if (typeof fn !== "function") continue;
      const node = state.groupNodes.get(groupId);
      if (!node) continue;
      try { fn(node, state); }
      catch (e) { console.warn(`form_builder hook ${groupId} failed`, e); }
    }
  }

  /* ----- commit + change notification ---------------------------------- */

  function _commit(state, field, value) {
    _setPath(state.config, field.path, value);
    state.onChange(field.path, value, state.config);
  }

  function _teardown(state) {
    state.host.innerHTML = "";
    state.groupNodes.clear();
    state.listeners.length = 0;
  }

  /* ----- path utilities ------------------------------------------------ */

  function _getPath(obj, path) {
    const parts = path.split(".");
    let cur = obj;
    for (const p of parts) {
      if (cur == null) return undefined;
      cur = cur[p];
    }
    return cur;
  }

  function _setPath(obj, path, value) {
    const parts = path.split(".");
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      const p = parts[i];
      if (cur[p] == null || typeof cur[p] !== "object") cur[p] = {};
      cur = cur[p];
    }
    if (value === null || value === undefined) {
      // Optional fields get cleared rather than left as null so the
      // emitted YAML stays clean.
      delete cur[parts[parts.length - 1]];
    } else {
      cur[parts[parts.length - 1]] = value;
    }
  }

  function _deepClone(v) {
    if (v == null) return v;
    try { return JSON.parse(JSON.stringify(v)); }
    catch (_) { return v; }
  }
})();
