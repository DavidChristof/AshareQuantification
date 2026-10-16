# 看板「加载失败，请检查 api 服务」—— 真凶是 CDN，不是服务

**日期**：2026-10-16
**起因**：用户报「api 服务为什么挂不起来」；追问后明确为**看板点「立即刷新」弹
「加载失败：…请确认 API 服务已启动」**。
**结论**：**API 服务从头到尾都是好的。** 真凶是 `frontend/index.html` 从
`cdn.jsdelivr.net` 拉 ECharts，而**本机连不上该域名**（curl `HTTP 000`）
=> `echarts` 未定义 => `loadAll()` 里那句 `drawManualCharts()` 抛 `ReferenceError`
=> 被同一个 `catch` 抓住 => 弹出那句把服务端冤枉了的提示。
**已修**：依赖本地化 + 错误提示按类型分开 + 缺库不再带崩整页 + 新增 `/vendor/{name}` 路由。

---

## 1. 先把服务端逐层排除（结论：全绿）

| 检查 | 结果 |
|---|---|
| 端口 / 进程 | LISTENING，进程活着 |
| 「立即刷新」调的**全部 11 个接口** | **全 200**（`/api/advisor` 0.18s、`/api/portfolio` 0.77s） |
| CORS（`file://` 页面即 `Origin: null`） | `allow_origins=["*"]`，实测回 `access-control-allow-origin: *` |
| 实时数据是否还在动 | 在动（601138 八秒内 48.05 -> 48.04，10 秒轮询线程活着） |
| 系统代理 / Clash | `ProxyEnable=0`、WinHTTP 直连、7890 未监听（**都不是**） |
| 当日崩溃转储 | 无（12:21 那个进程是被**终止**的，不是崩的） |

**服务端没有可查的了**，于是转向「浏览器 -> API 这一跳」。

## 2. 根因

`frontend/index.html` 头部两个依赖都走 CDN：

```html
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
```

实测（本机 curl）：

```
https://unpkg.com/vue@3/dist/vue.global.prod.js            HTTP 302  1.04s   <- 通
https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js  HTTP 000  0.24s   <- 连不上
```

于是链条（**代码路径逐行核对过，不是推测**）：

```
echarts.min.js 加载失败
  -> echarts 未定义
  -> loadAll() 里 this.drawManualCharts()          (在 try 内)
  -> drawManualCharts() -> drawMEquity() -> echarts.init(el)   (无任何守卫)
  -> ReferenceError: echarts is not defined
  -> catch(err) -> alert('加载失败：…请确认 API 服务已启动')   <- 把误诊写死
```

**Vue 能加载所以按钮存在、能点；点下去 11 个接口全部 200；然后在画图那一步炸掉。**
这就是「接口明明全通，却报服务没起来」的全部原因。

## 3. 放大器：那句 catch 把**任何**失败都说成服务故障

```js
// 修改前
alert('加载失败：' + err.message + '\n请确认 API 服务已启动（python api/main.py）');
```

它上面是一个 **11 个接口的 `Promise.all` + 后面的画图调用**，所以任何一环出错
（HTTP 500、网络抖动、**甚至前端自己的 JS 异常**）都会打印这句。**它是本次排查
最大的障碍** —— 一行"请确认 API 服务已启动"，把人往完全错误的方向带。

## 4. 修复（四处）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `frontend/vendor/`（新增） | **依赖本地化**：`vue.global.prod.js`（Vue **3.5.44**，170,605 B）、`echarts.min.js`（ECharts **5.6.1**，1,034,102 B），经 unpkg 下载并校验为真货（ECharts 带 `.echarts={}` UMD 全局、Vue 是顶层 `var Vue=`） |
| 2 | `frontend/index.html` 头部 | 两个 `<script src>` 改为**相对路径** `vendor/...`；页面里**再无任何 CDN 引用** |
| 3 | `frontend/index.html` 的 catch | 按错误类型分开说：**网络层失败**（`TypeError` / `Failed to fetch`）才提「连不上 API」；否则明说**这是页面内部的错误，不是 API 服务的问题** |
| 4 | `frontend/index.html::drawManualCharts` | 加 `typeof echarts === 'undefined'` 守卫：**缺库只该少几张图，不该让整页看起来像挂了** |
| 5 | `api/main.py`（新增路由） | `GET /vendor/{name}` -> `frontend/vendor/`，让**通过 `/dashboard` 打开**时同一行相对路径也能解析（原来 404）。`Path(name).name` 挡目录穿越 |

**第 5 条为什么必须**：页面里的 `vendor/xxx.js` 是相对路径 ——
用 `file://` 打开时指向本地文件（可用）；通过 `/dashboard` 打开时会解析成
`/vendor/xxx.js`，**没有这个路由就是 404，看板照样白屏**（实测过）。
两条打开方式现在都通。

## 5. 验证

```
py_compile api/main.py        OK
tests/test_gbk_output.py      4/4
全量 tests/test_*.py          31/31

重启后（新路由需重启才生效）：
  GET /dashboard             200  text/html  89,639 B
  GET /vendor/vue.global...  200  170,605 B   <- 与文件逐字节同尺寸
  GET /vendor/echarts.min.js 200  1,034,102 B <- 同
  GET /                      200  application/json   <- 未受影响（服务信息，非页面）
  GET /vendor/../api/main.py 404   <- 目录穿越被挡
  GET /vendor/nope.txt       404
```

**前置条件（重启前已核对）**：当日 09:31 调仓 marker 与流水线 marker 都还是 `2026-10-15`
（今天都还没跑），无 26/27/28/35 在跑；重启时刻 13:54 远早于 16:45/16:55 的窗口。

## 6. 遗留（如实记下）

1. **「页面能否渲染」这一层没有自动验证。** 试过用无头 Edge（`--headless=new --dump-dom`）
   加载页面，但**在本机跑不出来**（DOM 恒 0 字节、静态服务连一条请求都没收到），
   所以只能验到「页面所需的每个资源都以 200 + 正确尺寸可取」。
   **最终确认要靠人工：打开看板按 Ctrl+F5。**
2. `drawManualCharts` 之外的几处 `echarts.init`（tab 切换 988 行、分钟图 1071 行、
   净值/实盘图 1285/1425/1493/1514 行）**仍无守卫** —— 但它们不在 `loadAll` 的
   `try` 内，抛错不会误报"服务故障"，只会让对应那张图不出现。本轮**只修了会误导的那一处**。
3. **CDN 依赖要不要彻底禁掉**：本轮把这两个库本地化了，但如果以后有人往
   `index.html` 里再加一个 CDN `<script>`，同类故障会重演（本机对
   `cdn.jsdelivr.net` 不可达是**网络环境问题，不是临时的**）。约定：**看板依赖一律放
   `frontend/vendor/`**。
4. **`logs/app.log` 是 2026-10-08 之后才有的**（那次提交加的），所以更早的启动/运行
   记录只在 `logs/shadow_ab_<日期>.log` 与 Windows 事件日志里。另外
   `启动量化系统.bat` 的 `> logs\api.log 2>&1` 对子进程**无效**（api.log 恒 0 字节），
   排查时别拿它当依据 —— 用 `Start-Process -RedirectStandardError logs\api.err.log`。
5. **同一个坑的另一种形态**：如果今后 `frontend/vendor/` 里的文件被删/改名，
   页面会**静默**失去图和 Vue（后者更糟 —— 连按钮都没有）。第 4 条那个守卫至少会在
   Console 留一行 `echarts 未加载`。

## 相关

- `frontend/index.html`、`frontend/vendor/`、`api/main.py::dashboard_page` / `vendor_asset`
- `docs/2026-09-11-shadow-retry.md` —— 另一处"失败信号被吞掉/误导"的教训
- 本次会话另一条线：`docs/2026-09-24-manual-equity-missing-position.md`
