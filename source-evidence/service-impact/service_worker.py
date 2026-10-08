PROTOCOL_TEXT='{\n  "title": "Controlled Pi service-impact severity validation",\n  "version": "1.0",\n  "frozen_utc": "2026-10-08T00:10:56.781282+00:00",\n  "status": "Prospective revision-stage controlled experiment. Rules and schedule fixed before service outcomes. This is not original preregistration, independent timestamp certification, or a change to the original thirteen targets.",\n  "scope": "Validate temporal prediction of measured service-impact tiers for a disposable Pi loopback service. This does not validate LLM attack severity, industrial harm, human urgency judgments or automatic mitigation.",\n  "authorization": "User selected isolated Pi 5 service test in this conversation.",\n  "service": "New ephemeral Python standard-library HTTP service bound only to 127.0.0.1 on an OS-selected port. Echoes a nonce and its SHA-256 digest. No remote files, credentials, deployment changes, model calls or control actions.",\n  "request_slo_ms": 250,\n  "client_timeout_seconds": 0.75,\n  "rate_per_second": 8,\n  "max_client_workers": 4,\n  "max_server_workers": 8,\n  "monitor_requests_per_episode": 16,\n  "future_client_requests_per_episode": 24,\n  "recovery_requests_per_episode": 8,\n  "randomization_seed": 2026100802,\n  "conditions": [\n    {\n      "id": "healthy",\n      "monitor": "healthy",\n      "future": "healthy"\n    },\n    {\n      "id": "intermittent_delay",\n      "monitor": "delay20",\n      "future": "delay20"\n    },\n    {\n      "id": "intermittent_unavailable",\n      "monitor": "error20",\n      "future": "error20"\n    },\n    {\n      "id": "sustained_delay",\n      "monitor": "slow_all",\n      "future": "slow_all"\n    },\n    {\n      "id": "incorrect_success_body",\n      "monitor": "corrupt25",\n      "future": "corrupt25"\n    },\n    {\n      "id": "abrupt_onset",\n      "monitor": "healthy",\n      "future": "slow_all"\n    },\n    {\n      "id": "recovery_transition",\n      "monitor": "slow_all",\n      "future": "healthy"\n    }\n  ],\n  "episode_order": [\n    {\n      "episode": 1,\n      "block": 1,\n      "id": "intermittent_delay",\n      "monitor": "delay20",\n      "future": "delay20"\n    },\n    {\n      "episode": 2,\n      "block": 1,\n      "id": "recovery_transition",\n      "monitor": "slow_all",\n      "future": "healthy"\n    },\n    {\n      "episode": 3,\n      "block": 1,\n      "id": "incorrect_success_body",\n      "monitor": "corrupt25",\n      "future": "corrupt25"\n    },\n    {\n      "episode": 4,\n      "block": 1,\n      "id": "intermittent_unavailable",\n      "monitor": "error20",\n      "future": "error20"\n    },\n    {\n      "episode": 5,\n      "block": 1,\n      "id": "healthy",\n      "monitor": "healthy",\n      "future": "healthy"\n    },\n    {\n      "episode": 6,\n      "block": 1,\n      "id": "sustained_delay",\n      "monitor": "slow_all",\n      "future": "slow_all"\n    },\n    {\n      "episode": 7,\n      "block": 1,\n      "id": "abrupt_onset",\n      "monitor": "healthy",\n      "future": "slow_all"\n    },\n    {\n      "episode": 8,\n      "block": 2,\n      "id": "sustained_delay",\n      "monitor": "slow_all",\n      "future": "slow_all"\n    },\n    {\n      "episode": 9,\n      "block": 2,\n      "id": "incorrect_success_body",\n      "monitor": "corrupt25",\n      "future": "corrupt25"\n    },\n    {\n      "episode": 10,\n      "block": 2,\n      "id": "healthy",\n      "monitor": "healthy",\n      "future": "healthy"\n    },\n    {\n      "episode": 11,\n      "block": 2,\n      "id": "abrupt_onset",\n      "monitor": "healthy",\n      "future": "slow_all"\n    },\n    {\n      "episode": 12,\n      "block": 2,\n      "id": "recovery_transition",\n      "monitor": "slow_all",\n      "future": "healthy"\n    },\n    {\n      "episode": 13,\n      "block": 2,\n      "id": "intermittent_delay",\n      "monitor": "delay20",\n      "future": "delay20"\n    },\n    {\n      "episode": 14,\n      "block": 2,\n      "id": "intermittent_unavailable",\n      "monitor": "error20",\n      "future": "error20"\n    },\n    {\n      "episode": 15,\n      "block": 3,\n      "id": "incorrect_success_body",\n      "monitor": "corrupt25",\n      "future": "corrupt25"\n    },\n    {\n      "episode": 16,\n      "block": 3,\n      "id": "intermittent_delay",\n      "monitor": "delay20",\n      "future": "delay20"\n    },\n    {\n      "episode": 17,\n      "block": 3,\n      "id": "recovery_transition",\n      "monitor": "slow_all",\n      "future": "healthy"\n    },\n    {\n      "episode": 18,\n      "block": 3,\n      "id": "healthy",\n      "monitor": "healthy",\n      "future": "healthy"\n    },\n    {\n      "episode": 19,\n      "block": 3,\n      "id": "abrupt_onset",\n      "monitor": "healthy",\n      "future": "slow_all"\n    },\n    {\n      "episode": 20,\n      "block": 3,\n      "id": "sustained_delay",\n      "monitor": "slow_all",\n      "future": "slow_all"\n    },\n    {\n      "episode": 21,\n      "block": 3,\n      "id": "intermittent_unavailable",\n      "monitor": "error20",\n      "future": "error20"\n    },\n    {\n      "episode": 22,\n      "block": 4,\n      "id": "abrupt_onset",\n      "monitor": "healthy",\n      "future": "slow_all"\n    },\n    {\n      "episode": 23,\n      "block": 4,\n      "id": "recovery_transition",\n      "monitor": "slow_all",\n      "future": "healthy"\n    },\n    {\n      "episode": 24,\n      "block": 4,\n      "id": "intermittent_delay",\n      "monitor": "delay20",\n      "future": "delay20"\n    },\n    {\n      "episode": 25,\n      "block": 4,\n      "id": "intermittent_unavailable",\n      "monitor": "error20",\n      "future": "error20"\n    },\n    {\n      "episode": 26,\n      "block": 4,\n      "id": "healthy",\n      "monitor": "healthy",\n      "future": "healthy"\n    },\n    {\n      "episode": 27,\n      "block": 4,\n      "id": "sustained_delay",\n      "monitor": "slow_all",\n      "future": "slow_all"\n    },\n    {\n      "episode": 28,\n      "block": 4,\n      "id": "incorrect_success_body",\n      "monitor": "corrupt25",\n      "future": "corrupt25"\n    }\n  ],\n  "faults": {\n    "healthy": "Correct HTTP 200 nonce/digest response after 2ms authored delay.",\n    "delay20": "Each request independently has probability0.20 of 300ms authored delay; otherwise2ms.",\n    "error20": "Each request independently has probability0.20 of HTTP503; otherwise correct200 after2ms.",\n    "slow_all": "Every response receives450ms authored delay, with correctHTTP200 body.",\n    "corrupt25": "Each request independently has probability0.25 of incorrect digest despiteHTTP200, otherwise correct; no actual storeddata is altered."\n  },\n  "fault_randomization": "Each request uses an independent pseudorandom draw keyed by fixed seed, episode, phase and unique nonce; monitor and future requests use different seeds/nonces. Fault configuration never enters either prediction function or outcome labeling.",\n  "episode_procedure": [\n    "Run16 monitorrequests at8/s; wait for all results.",\n    "Commit both forecast tiers using only those monitor observations before initiating any later client request.",\n    "Run24 different later clientrequests at8/s. Compare observed correct completion within250ms against the committed forecasts.",\n    "Restore only this disposable service to healthy. Collect8 additional recoveryrequests after draining previous requests.",\n    "Retain all raw request records, timestamps, predictions, faultsettings and observed outcomes; randomize seven conditions in each of four blocks."\n  ],\n  "primary_predictor": "Functional-monitor persistence forecast: count earlier requests not completed withHTTP200, matchingnonce and correctdigest within250ms; map that observed fraction to a tier. Inputs contain earlier observations only, with no future data or injected condition.",\n  "baseline_predictor": "Transport-only persistence forecast: count earlier requests not completed withHTTP200 within250ms; ignore functionalbody correctness. Same tierthresholds and timing.",\n  "reference": "The later independent24request clientwindow determines observed impact tier from its actual fraction of failed functionalSLO requests. Fault schedule/probability/expected counts are never used as groundtruth. HTTP status, body correctness, elapsedwalltime and exceptions are retained.",\n  "tiers": {\n    "None": "0 failed functionalSLO requests.",\n    "Low": "More than0 and at most10% fail.",\n    "Moderate": "More than10% and at most30% fail.",\n    "High": "More than30% fail."\n  },\n  "tier_scope": "Authored serviceSLO impact categories for this experimental endpoint; no standards endorsement or measured physical/safety consequence. Monitor16 and future24 requests differ; the two tiers can disagree from sampling, drift, onset or recovery.",\n  "primary_metrics": [\n    "4x4 confusion matrix; exact forecastagreement count/28 for each predictor",\n    "under-triage and over-triage counts by ordinaltiers",\n    "missedHigh outcomes and Highrecall denominator",\n    "paired comparison on incorrect-success-body challenge",\n    "percondition and perblock records; recoverySLO counts"\n  ],\n  "reporting": "Retain every episode, exception and outcome. No parameter fitting, outcome-driven reruns, selection of successful episodes or post-result threshold changes. Report exact counts, not broad industrial generalization. Distinguish stationaryfiveconditions from two preregistered transitionchallenges, reportboth andtotal.",\n  "failure_policy": "If technical failure prevents complete collection, report all emitted data as an interrupted experiment. Do not silently replace the run. Request-level failures/timeouts are valid outcomes and must be retained.",\n  "resource_bounds": "At most8 launchedrequests/s,4clientworkers and8serverworkers; tiny bodies, delays<=450ms, total1344 plannedrequests, hardworker watchdog240s. UseOSwallclock measurements; no loadflooding, packetspoofing or contact withother endpoints. Servercloses andallworkerthreadsjoin in finally.",\n  "evidence": [\n    "PROTOCOL.json and SHA-256",\n    "worker source and SHA-256",\n    "raw JSONL eventstream including all requests and forecast-before-outcome timestamps",\n    "execution receipt/device metadata",\n    "independent arithmetic and scientific audits",\n    "paper/redline/reviewerresponse and R5 evidencearchive"\n  ]\n}'
BODY_SHA256='aa858767ebe35ec12c7645bbffdb87b7fc55a78fef5d16e071003ba591213b3f'
import concurrent.futures,datetime,hashlib,http.client,http.server,json,os,platform,random,socket,sys,threading,time,urllib.parse,resource

P=json.loads(PROTOCOL_TEXT)
START=time.perf_counter_ns();LOCK=threading.Lock();EMIT_LOCK=threading.Lock()
CONTEXT={'mode':'healthy','episode':0,'phase':'startup'}
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def emit(kind,**data):
    with EMIT_LOCK:
        print(json.dumps({'event':kind,'utc':stamp(),'elapsed_ns':time.perf_counter_ns()-START,**data},sort_keys=True,separators=(',',':')),flush=True)
def watchdog():
    emit('aborted',reason='240-second hard watchdog');os._exit(99)
timer=threading.Timer(240,watchdog);timer.daemon=True;timer.start()

class Server(http.server.ThreadingHTTPServer):
    daemon_threads=False
    block_on_close=True
    request_queue_size=8
    allow_reuse_address=False
    def __init__(self,*a,**k):
        self.slots=threading.BoundedSemaphore(P['max_server_workers']);super().__init__(*a,**k)
    def process_request(self,request,client_address):
        self.slots.acquire()
        try:super().process_request(request,client_address)
        except BaseException:self.slots.release();raise
    def process_request_thread(self,request,client_address):
        try:super().process_request_thread(request,client_address)
        finally:self.slots.release()

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def do_GET(self):
        n=urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get('n',[''])[0]
        if len(n)!=32 or any(c not in '0123456789abcdef' for c in n):
            self.send_error(400);return
        with LOCK:c=dict(CONTEXT)
        seed=hashlib.sha256(f"{P['randomization_seed']}|{c['episode']}|{c['phase']}|{n}|fault".encode()).digest()
        u=random.Random(int.from_bytes(seed,'big')).random()
        delay=.002;status=200;digest=hashlib.sha256((n+'|service-v1').encode()).hexdigest()
        if c['mode']=='delay20' and u<.2:delay=.300
        if c['mode']=='error20' and u<.2:status=503
        if c['mode']=='slow_all':delay=.450
        if c['mode']=='corrupt25' and u<.25:digest='0'*64
        time.sleep(delay)
        body=json.dumps({'nonce':n,'digest':digest},separators=(',',':')).encode()
        try:
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

def request(port,ep,phase,i):
    nonce=hashlib.sha256(f"{P['randomization_seed']}|{ep}|{phase}|{i}|client".encode()).hexdigest()[:32]
    expected=hashlib.sha256((nonce+'|service-v1').encode()).hexdigest()
    t=time.perf_counter_ns();started=stamp();status=None;correct=False;error=None;body=b'';got_nonce=None;got_digest=None
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=P['client_timeout_seconds'])
    try:
        conn.request('GET','/work?n='+nonce,headers={'Connection':'close'})
        response=conn.getresponse();status=response.status;body=response.read(2048)
        try:
            d=json.loads(body);got_nonce=d.get('nonce');got_digest=d.get('digest');correct=got_nonce==nonce and got_digest==expected
        except (ValueError,AttributeError,TypeError):correct=False
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    finally:conn.close()
    elapsed=time.perf_counter_ns()-t
    transport=status==200 and error is None and elapsed<=P['request_slo_ms']*1000000
    out={'episode':ep,'phase':phase,'request_index':i,'nonce':nonce,'started_utc':started,'started_elapsed_ns':t-START,'finished_elapsed_ns':time.perf_counter_ns()-START,'duration_ns':elapsed,'status':status,'body_correct':correct,'returned_nonce':got_nonce,'returned_digest':got_digest,'response_sha256':hashlib.sha256(body).hexdigest(),'error':error,'transport_slo_success':transport,'functional_slo_success':transport and correct}
    emit('request',**out);return out

def window(pool,port,ep,phase,mode,n):
    with LOCK:CONTEXT.update(mode=mode,episode=ep,phase=phase)
    futures=[];base=time.perf_counter()
    for i in range(n):
        wait=base+i/P['rate_per_second']-time.perf_counter()
        if wait>0:time.sleep(wait)
        futures.append(pool.submit(request,port,ep,phase,i))
    return [f.result() for f in futures]

def tier(bad,n):
    if bad==0:return 'None'
    if 10*bad<=n:return 'Low'
    if 10*bad<=3*n:return 'Moderate'
    return 'High'

def predict(early):
    # This function receives only earlier observations, never condition/future outcomes.
    functional_bad=sum(not x['functional_slo_success'] for x in early)
    transport_bad=sum(not x['transport_slo_success'] for x in early)
    return {'functional':tier(functional_bad,len(early)),'transport_only':tier(transport_bad,len(early)),'functional_bad':functional_bad,'transport_bad':transport_bad,'n':len(early)}

server=None;thread=None;pool=None;completed=False
try:
    server=Server(('127.0.0.1',0),Handler);port=server.server_address[1]
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05},name='isolated-http-service',daemon=True);thread.start()
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=P['max_client_workers'],thread_name_prefix='isolated-client')
    emit('run_started',hostname=platform.node(),machine=platform.machine(),python=sys.version,pid=os.getpid(),protocol_sha256=hashlib.sha256(PROTOCOL_TEXT.encode()).hexdigest(),body_sha256=BODY_SHA256,endpoint='127.0.0.1:'+str(port),configured_rate_per_second=P['rate_per_second'],client_workers=P['max_client_workers'],server_workers=P['max_server_workers'],load_average=os.getloadavg())
    for ep in P['episode_order']:
        eid=ep['episode'];emit('episode_started',episode=eid,block=ep['block'],condition=ep['id'])
        early=window(pool,port,eid,'monitor',ep['monitor'],P['monitor_requests_per_episode'])
        forecast=predict(early)
        early_hash=hashlib.sha256(json.dumps(early,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        emit('forecast_committed',episode=eid,forecast=forecast,early_records_sha256=early_hash)
        future=window(pool,port,eid,'future_client',ep['future'],P['future_client_requests_per_episode'])
        bad=sum(not x['functional_slo_success'] for x in future)
        observed=tier(bad,len(future))
        recovery=window(pool,port,eid,'recovery','healthy',P['recovery_requests_per_episode'])
        emit('episode_completed',episode=eid,block=ep['block'],condition=ep['id'],forecast=forecast,observed_tier=observed,future_functional_bad=bad,future_n=len(future),recovery_bad=sum(not x['functional_slo_success'] for x in recovery),recovery_n=len(recovery))
    completed=True
finally:
    if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
    if server is not None:server.shutdown();server.server_close()
    if thread is not None:thread.join(timeout=2)
    timer.cancel()
    emit('run_closed',complete=completed,service_thread_alive=thread.is_alive() if thread else False,load_average=os.getloadavg(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
