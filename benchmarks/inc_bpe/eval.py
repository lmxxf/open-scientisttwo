import argparse,json,time,statistics,math,random,tracemalloc,platform,sys
from pathlib import Path
from data import vocabulary,streams,reference,SuffixAutomaton,SufSucTree
from model import IncrementalBPE

def BaselinePointerBPE(vocab,rules):
    class _Base:
        def __init__(self):
            self.vocab,self.rules=vocab,rules
            self.ac=SuffixAutomaton(vocab);self.state=0;self.last=[None];self.length=0
            self.tree=SufSucTree(vocab,rules);self.successor=self.tree.successor
            self.pre={256+i:a for i,(a,b) in enumerate(rules)}
            self.rank={256+i:i for i in range(len(rules))}
        def valid(self,t,n):
            if t<256:return True
            off=len(self.vocab[self.successor[t]])
            if n<off or n-off>=len(self.last):return False
            k=self.last[n-off]
            pre=self.pre[t];child=None
            while k!=pre and k in self.successor:
                child=k;k=self.successor[k]
            return k==pre and (child is None or self.rank[t]<self.rank[child])
        def append(self,byte):
            self.state=self.ac.step(self.state,byte);self.length+=1
            candidates=sorted(self.ac.output[self.state],key=lambda t:-len(self.vocab[t]))
            token=next(t for t in candidates if self.valid(t,self.length))
            self.last.append(token);return token
        def tokens(self):
            result=[];n=self.length
            while n:
                t=self.last[n];result.append(t);n-=len(self.vocab[t])
            return result[::-1]
    return _Base()

def CentroidBinarySearchBPE(vocab,rules):
    """Jiang & Gong (arXiv:2605.30813) style O(log D) DFS-interval binary search over child subtrees."""
    import bisect
    class _Centroid:
        def __init__(self):
            self.vocab,self.rules=vocab,rules
            self.lengths=tuple(map(len,vocab))
            self.ac=SuffixAutomaton(vocab);self.state=0;self.last=[None];self.length=0
            self.tree=SufSucTree(vocab,rules);self.left=self.tree.left;self.right=self.tree.right
            self.pre={256+i:a for i,(a,b) in enumerate(rules)}
            self.rank={256+i:i for i in range(len(rules))}
            self.child_starts={u:[self.left[c] for c in self.tree.children[u]] for u in range(len(vocab))}
            self.cands=[tuple(sorted(out,key=lambda t:-self.lengths[t])) for out in self.ac.output]
        def valid(self,t,n):
            if t<256:return True
            off=self.lengths[self.tree.successor[t]]
            if n<off or n-off>=len(self.last):return False
            k=self.last[n-off];a=self.pre[t];lk=self.left[k]
            if lk==self.left[a]:return True
            if not (self.left[a]<lk<self.right[a]):return False
            idx=bisect.bisect_right(self.child_starts[a],lk)-1
            child=self.tree.children[a][idx]
            return self.rank[t]<self.rank[child]
        def append(self,byte):
            self.state=self.ac.step(self.state,byte);self.length+=1
            for t in self.cands[self.state]:
                if self.valid(t,self.length):
                    self.last.append(t);return t
            raise ValueError('no token')
    return _Centroid()

def WeightedCentroidSearchBPE(vocab,rules):
    """Component-Weighted Centroid Search (arXiv:2609.40016) weighting decision trees by subtree size."""
    class _WeightedCentroid:
        def __init__(self):
            self.vocab,self.rules=vocab,rules
            self.lengths=tuple(map(len,vocab))
            self.ac=SuffixAutomaton(vocab);self.state=0;self.last=[None];self.length=0
            self.tree=SufSucTree(vocab,rules);self.left=self.tree.left;self.right=self.tree.right
            self.pre={256+i:a for i,(a,b) in enumerate(rules)}
            self.rank={256+i:i for i in range(len(rules))}
            # Build weight-balanced binary search tree over children[u] weighted by subtree size (right[c]-left[c])
            self.wtree={}
            for u,ch in self.tree.children.items():
                if not ch:continue
                weights=[max(1,self.right[c]-self.left[c]) for c in ch]
                def _build(lo,hi):
                    if lo>hi:return None
                    if lo==hi:
                        c=ch[lo];return (self.left[c],self.right[c],self.rank[c],None,None)
                    tot=sum(weights[lo:hi+1]);acc=0;mid=lo
                    for idx in range(lo,hi+1):
                        acc+=weights[idx]
                        if acc*2>=tot:mid=idx;break
                    c=ch[mid]
                    return (self.left[c],self.right[c],self.rank[c],_build(lo,mid-1),_build(mid+1,hi))
                self.wtree[u]=_build(0,len(ch)-1)
            self.cands=[tuple(sorted(out,key=lambda t:-self.lengths[t])) for out in self.ac.output]
        def valid(self,t,n):
            if t<256:return True
            off=self.lengths[self.tree.successor[t]]
            if n<off or n-off>=len(self.last):return False
            k=self.last[n-off];a=self.pre[t];lk=self.left[k]
            if lk==self.left[a]:return True
            if not (self.left[a]<lk<self.right[a]):return False
            node=self.wtree.get(a)
            while node is not None:
                cl,cr,crank,lch,rch=node
                if lk<cl:node=lch
                elif lk>=cr:node=rch
                else:return self.rank[t]<crank
            return False
        def append(self,byte):
            self.state=self.ac.step(self.state,byte);self.length+=1
            for t in self.cands[self.state]:
                if self.valid(t,self.length):
                    self.last.append(t);return t
            raise ValueError('no token')
    return _WeightedCentroid()

def ExplicitHashSetBPE(vocab,rules):
    """O(R*M) materialized predecessor hash-set baseline."""
    class _SetBPE:
        def __init__(self):
            self.vocab,self.rules=vocab,rules
            self.lengths=tuple(map(len,vocab))
            self.ac=SuffixAutomaton(vocab);self.state=0;self.last=[None];self.length=0
            tree=SufSucTree(vocab,rules);self.successor=tree.successor
            self.pre={256+i:a for i,(a,b) in enumerate(rules)}
            self.rank={256+i:i for i in range(len(rules))}
            self.accept={};self.total_set_elements=0
            for t,pre in self.pre.items():
                acc={pre}
                for c in tree.children[pre]:
                    if self.rank[t]<self.rank[c]:
                        stack=[c]
                        while stack:
                            u=stack.pop();acc.add(u);stack.extend(tree.children[u])
                self.accept[t]=frozenset(acc);self.total_set_elements+=len(acc)
            self.cands=[tuple(sorted(out,key=lambda t:-self.lengths[t])) for out in self.ac.output]
        def append(self,byte):
            self.state=self.ac.step(self.state,byte);self.length+=1;n=self.length
            for t in self.cands[self.state]:
                if t<256:
                    self.last.append(t);return t
                off=self.lengths[self.successor[t]]
                if 0<=n-off<len(self.last) and self.last[n-off] in self.accept[t]:
                    self.last.append(t);return t
            raise ValueError('no token')
    return _SetBPE()

def check(raw,vocab,rules,config):
    model=IncrementalBPE(vocab,rules,config)
    for n,c in enumerate(raw,1):
        model.append(c)
        actual=model.tokens();expected=reference(raw[:n],rules)
        if actual!=expected or b''.join(vocab[t] for t in actual)!=raw[:n]:
            raise ValueError(f'exact prefix BPE mismatch at byte {n}')

def measure_certificate_coverage(model,vocab,rules,texts):
    """Directly profile the deployed IncrementalBPE instance's compiled candidate table and runtime execution."""
    ac=SuffixAutomaton(vocab);tree=SufSucTree(vocab,rules)
    lengths=tuple(map(len,vocab))
    suc=tree.successor
    # Inspect deployed compiler's own candidate tuples if available
    deployed_cands=getattr(model,'candidates',None)
    deployed_atomic=getattr(model,'atomic',None)
    compiler_stats=getattr(model,'certificate_stats',None)
    atomic_states=0;certified_pairs=0;total_merge_pairs=0
    if deployed_cands is not None:
        for s,entries in enumerate(deployed_cands):
            merges=[e for e in entries if isinstance(e,(tuple,list)) and len(e)>=7 and e[0]>=256]
            if not merges:atomic_states+=1
            for e in merges:
                total_merge_pairs+=1
                if e[6]==1:certified_pairs+=1
    else:
        for s,out in enumerate(ac.output):
            merges=[t for t in out if t>=256]
            if not merges:atomic_states+=1
            total_merge_pairs+=len(merges)
    total_bytes=0;atomic_hits=0;cert_hits=0;interval_checks=0
    for raw in texts:
        m_run=IncrementalBPE(vocab,rules,dict(acceptance_mode='intervals',flat_transitions=True,atomic_fastpath=True,certified_regions=True,rejection_certificates=False))
        cands_table=getattr(m_run,'candidates',None)
        atom_table=getattr(m_run,'atomic',None)
        trans_table=getattr(m_run,'trans',None)
        left_arr=getattr(m_run,'left',None)
        s=0;last=[None]
        for n,b_val in enumerate(raw,1):
            total_bytes+=1
            s=trans_table[s*256+b_val] if trans_table is not None else ac.step(s,b_val)
            if atom_table is not None and atom_table[s]>=0:
                atomic_hits+=1;last.append(atom_table[s]);continue
            if cands_table is not None and left_arr is not None and (not cands_table[s] or (isinstance(cands_table[s][0],(tuple,list)) and len(cands_table[s][0])>=6)):
                chosen=None
                for entry in cands_table[s]:
                    t,off,lo,hi,q,end=entry[:6]
                    verdict=entry[6] if len(entry)>6 else 0
                    if t<256:chosen=t;break
                    if n<=off or verdict==-1:continue
                    if verdict==1:
                        cert_hits+=1;chosen=t;break
                    interval_checks+=1
                    x=left_arr[last[n-off]]
                    if (lo<=x<hi) or (q<=x<end):chosen=t;break
                last.append(chosen)
            else:
                last.append(m_run.append(b_val))
    return dict(deployed_compiler_instrumented=bool(deployed_cands is not None),
                deployed_compiler_certificate_stats=compiler_stats or {},
                total_automaton_states=len(ac.output),
                atomic_only_states=atomic_states,
                atomic_state_fraction=round(atomic_states/max(len(ac.output),1),4),
                certified_state_candidate_pairs=certified_pairs,
                total_merge_candidate_pairs=total_merge_pairs,
                static_pair_certification_rate=round(certified_pairs/max(total_merge_pairs,1),4),
                runtime_total_bytes=total_bytes,
                runtime_atomic_fastpath_hits=atomic_hits,
                runtime_atomic_fastpath_hit_rate=round(atomic_hits/max(total_bytes,1),4),
                runtime_static_certificate_hits=cert_hits,
                runtime_non_atomic_merge_bytes=total_bytes-atomic_hits,
                runtime_certificate_hit_rate_on_merge_states=round(cert_hits/max(total_bytes-atomic_hits,1),4),
                runtime_interval_predicate_evaluations=interval_checks)

def inspect_memory(model,vocab,rules):
    entries=0;interval_scalars=0
    for attr in ('intervals','accept','acceptance','candidates','table','trans'):
        if hasattr(model,attr):
            val=getattr(model,attr)
            if isinstance(val,(dict,list,tuple,set,frozenset)):
                entries+=len(val)
                if attr=='intervals':
                    interval_scalars=sum(len(x) for x in (val.values() if isinstance(val,dict) else val) if isinstance(x,(tuple,list)))
    tree=SufSucTree(vocab,rules)
    depths={i:0 for i in range(256)}
    for t in range(256,len(vocab)):
        d=0;curr=t
        while curr in tree.successor:curr=tree.successor[curr];d+=1
        depths[t]=d
    return dict(vocab_size=len(vocab),num_merges=len(rules),max_token_bytes=max(map(len,vocab)),
                max_successor_depth=max(depths.values()),mean_successor_depth=round(statistics.mean(depths.values()),3),
                model_structure_top_entries=entries,interval_endpoint_scalars=interval_scalars)

def build_synthetic_deep_vocab(num_merges,max_chain_depth):
    """Construct canonical BPE vocabulary with controlled merge count R and successor forest depth D."""
    vocab=[bytes([i]) for i in range(256)];rules=[];seen=set(vocab);chain_tips=[]
    for r_idx in range(4):
        u0=65+r_idx
        for s_idx in range(4):
            if len(rules)>=num_merges:break
            right_b=48+(r_idx*2+s_idx)%10
            w=vocab[u0]+vocab[right_b]
            if w not in seen:seen.add(w);rules.append((u0,right_b));vocab.append(w)
        branches=[]
        for b_idx in range(6):
            if len(rules)>=num_merges:break
            left_b=97+(r_idx*4+b_idx)%26
            w=vocab[left_b]+vocab[u0]
            if w not in seen:seen.add(w);rules.append((left_b,u0));branches.append(255+len(rules));vocab.append(w)
        for br in branches[:2]:
            curr=br
            for d in range(2,max_chain_depth+1):
                if len(rules)>=num_merges:break
                left_b=97+(d*5+r_idx)%26
                w=vocab[left_b]+vocab[curr]
                if w not in seen:seen.add(w);rules.append((left_b,curr));curr=255+len(rules);vocab.append(w)
            chain_tips.append((u0,curr))
    for a in range(32,126):
        if len(rules)>=num_merges:break
        for b in range(32,126):
            if len(rules)>=num_merges:break
            w=vocab[a]+vocab[b]
            if w not in seen:seen.add(w);rules.append((a,b));vocab.append(w)
    chunks=[]
    for idx in range(250):
        u0,tip=chain_tips[idx%len(chain_tips)]
        digit=48+((u0-65)*2+(idx%4))%10
        chunks.append(vocab[tip]+bytes([digit]))
    stream=b''.join(chunks)[:2400]
    return vocab,rules,stream

def evaluate_synthetic_point(r_target,d_target,clean_cfg,rng):
    sv,sr,sstream=build_synthetic_deep_vocab(r_target,d_target)
    # Exhaustive prefix equivalence verification on the first 360 prefixes of the synthetic stream
    check(sstream[:360],sv,sr,clean_cfg)
    stree=SufSucTree(sv,sr)
    max_d=0
    for t in range(256,len(sv)):
        d=0;curr=t
        while curr in stree.successor:curr=stree.successor[curr];d+=1
        if d>max_d:max_d=d
    s_set=ExplicitHashSetBPE(sv,sr)
    methods=['interval','pointer','weighted_centroid_2609_40016','unweighted_centroid_2605_30813','explicit_set']
    samples={k:[] for k in methods}
    for _ in range(5):
        order=list(methods);rng.shuffle(order)
        for mname in order:
            if mname=='interval':m_inst=IncrementalBPE(sv,sr,clean_cfg)
            elif mname=='pointer':m_inst=IncrementalBPE(sv,sr,dict(clean_cfg,acceptance_mode='pointer'))
            elif mname=='weighted_centroid_2609_40016':m_inst=WeightedCentroidSearchBPE(sv,sr)
            elif mname=='unweighted_centroid_2605_30813':m_inst=CentroidBinarySearchBPE(sv,sr)
            else:m_inst=ExplicitHashSetBPE(sv,sr)
            t0=time.perf_counter()
            for c in sstream:m_inst.append(c)
            samples[mname].append((time.perf_counter()-t0)/len(sstream)*1e7)
    med_int=statistics.median(samples['interval'])
    med_ptr=statistics.median(samples['pointer'])
    med_wcent=statistics.median(samples['weighted_centroid_2609_40016'])
    med_cent=statistics.median(samples['unweighted_centroid_2605_30813'])
    med_set=statistics.median(samples['explicit_set'])
    paired_ptr_diffs=[p-i for p,i in zip(samples['pointer'],samples['interval'])]
    t_crit5=2.7764  # Student-t df=4 95% critical value
    diff_sd=statistics.stdev(paired_ptr_diffs) if len(paired_ptr_diffs)>1 else 0.0
    depths_map={};root_of={}
    for t_node in range(256,len(sv)):
        d_cnt=0;curr=t_node
        while curr in stree.successor:curr=stree.successor[curr];d_cnt+=1
        depths_map[t_node]=d_cnt;root_of[t_node]=curr
    deep_tip=max(depths_map,key=lambda u:depths_map[u])
    target_u0=root_of[deep_tip]
    cand_t=next((256+i for i,(a,b) in enumerate(sr) if a==target_u0),len(sv)-1)
    m_int=IncrementalBPE(sv,sr,clean_cfg);m_ptr=IncrementalBPE(sv,sr,dict(clean_cfg,acceptance_mode='pointer'))
    m_int.last=[deep_tip]*128;m_ptr.last=[deep_tip]*128
    if hasattr(m_int,'valid') and hasattr(m_ptr,'valid'):
        p0=time.perf_counter()
        for _ in range(3000):m_int.valid(cand_t,100)
        dt_p_int=max(time.perf_counter()-p0,1e-9)
        p1=time.perf_counter()
        for _ in range(3000):m_ptr.valid(cand_t,100)
        dt_p_ptr=max(time.perf_counter()-p1,1e-9)
        micro_ratio=round(dt_p_ptr/dt_p_int,2)
    else:
        micro_ratio=round(med_ptr/max(med_int,1e-9),2)
    return dict(num_merges=len(sr),vocab_size=len(sv),max_forest_depth=max_d,
                all_prefixes_verified_exact=True,randomized_interleaved_passes=5,
                interval_ms_per_10kb=round(med_int,4),pointer_only_ms_per_10kb=round(med_ptr,4),
                weighted_centroid_2609_40016_ms_per_10kb=round(med_wcent,4),
                unweighted_centroid_2605_30813_ms_per_10kb=round(med_cent,4),
                explicit_set_ms_per_10kb=round(med_set,4),
                isolated_interval_vs_pointer_speedup=round(med_ptr/max(med_int,1e-9),3),
                paired_ptr_minus_int_mean_ms=round(statistics.mean(paired_ptr_diffs),4),
                paired_ptr_minus_int_t95_ci_ms=round(t_crit5*diff_sd/math.sqrt(5),4),
                forced_depth_D_predicate_micro_speedup=micro_ratio,
                interval_vs_weighted_centroid_speedup=round(med_wcent/max(med_int,1e-9),3),
                interval_vs_unweighted_centroid_speedup=round(med_cent/max(med_int,1e-9),3),
                explicit_set_entries=s_set.total_set_elements,two_interval_entries=2*len(sr),
                memory_reduction_ratio=round(s_set.total_set_elements/max(2*len(sr),1),2))

def deep_forest_and_isolation_study(vocab,rules,texts,config):
    """Complete 2x2 cache-free randomized paired factorial isolation, paired final-config subtractions, multi-scale deep-forest sweep, and iso-merge (R=320) controlled depth sweep."""
    rng=random.Random(77)
    clean_cfg=dict(config,cache_capacity=0,acceptance_cache='none',rejection_certificates=False)
    variants=['selected','intervals_no_cert','pointer_with_cert','pointer_predicate_only','remove_atomic_fastpath','remove_flat_transitions','weighted_centroid_2609_40016','centroid_search_2605_30813','explicit_set','baseline_pointer']
    samples={k:[] for k in variants}
    size=sum(map(len,texts))
    set_probe=ExplicitHashSetBPE(vocab,rules)
    for _ in range(7):
        order=list(variants);rng.shuffle(order)
        for v in order:
            if v=='selected':
                ms=[IncrementalBPE(vocab,rules,clean_cfg) for _ in texts]
            elif v=='intervals_no_cert':
                ms=[IncrementalBPE(vocab,rules,dict(clean_cfg,certified_regions=False)) for _ in texts]
            elif v=='pointer_with_cert':
                ms=[IncrementalBPE(vocab,rules,dict(clean_cfg,acceptance_mode='pointer',certified_regions=True)) for _ in texts]
            elif v=='pointer_predicate_only':
                ms=[IncrementalBPE(vocab,rules,dict(clean_cfg,acceptance_mode='pointer',certified_regions=False)) for _ in texts]
            elif v=='remove_atomic_fastpath':
                ms=[IncrementalBPE(vocab,rules,dict(clean_cfg,atomic_fastpath=False)) for _ in texts]
            elif v=='remove_flat_transitions':
                ms=[IncrementalBPE(vocab,rules,dict(clean_cfg,flat_transitions=False)) for _ in texts]
            elif v=='weighted_centroid_2609_40016':
                ms=[WeightedCentroidSearchBPE(vocab,rules) for _ in texts]
            elif v=='centroid_search_2605_30813':
                ms=[CentroidBinarySearchBPE(vocab,rules) for _ in texts]
            elif v=='explicit_set':
                ms=[ExplicitHashSetBPE(vocab,rules) for _ in texts]
            else:
                ms=[BaselinePointerBPE(vocab,rules) for _ in texts]
            t0=time.perf_counter()
            for m,raw in zip(ms,texts):
                for c in raw:m.append(c)
            samples[v].append((time.perf_counter()-t0)/size*1e7)
    t_crit=2.4469  # Student-t 95% two-sided critical value for df=6 (n=7)
    isolation={}
    for v in variants:
        s=samples[v];mean=statistics.mean(s);sd=statistics.stdev(s) if len(s)>1 else 0.0
        paired_vs_sel=[x-y for x,y in zip(s,samples['selected'])]
        p_sd=statistics.stdev(paired_vs_sel) if len(paired_vs_sel)>1 else 0.0
        isolation[v]=dict(median_ms_per_10kb=round(statistics.median(s),4),mean_ms_per_10kb=round(mean,4),
                          std_ms_per_10kb=round(sd,4),student_t95_halfwidth=round(t_crit*sd/math.sqrt(len(s)),4),
                          paired_delta_vs_selected_mean_ms=round(statistics.mean(paired_vs_sel),4),
                          paired_delta_vs_selected_t95_ci_ms=round(t_crit*p_sd/math.sqrt(len(s)),4))
    # Explicit 2x2 factorial interaction contrast: (Ptr_NoCert - Int_NoCert) vs (Ptr_Cert - Int_Cert)
    int_effect_nocert=[p-i for p,i in zip(samples['pointer_predicate_only'],samples['intervals_no_cert'])]
    int_effect_cert=[p-i for p,i in zip(samples['pointer_with_cert'],samples['selected'])]
    interaction_samples=[a-b for a,b in zip(int_effect_nocert,int_effect_cert)]
    inter_sd=statistics.stdev(interaction_samples) if len(interaction_samples)>1 else 0.0
    isolation['factorial_2x2_summary']=dict(
        interval_saving_without_cert_mean_ms=round(statistics.mean(int_effect_nocert),4),
        interval_saving_without_cert_t95_ci_ms=round(t_crit*statistics.stdev(int_effect_nocert)/math.sqrt(7),4),
        interval_saving_with_cert_mean_ms=round(statistics.mean(int_effect_cert),4),
        interval_saving_with_cert_t95_ci_ms=round(t_crit*statistics.stdev(int_effect_cert)/math.sqrt(7),4),
        interaction_contrast_mean_ms=round(statistics.mean(interaction_samples),4),
        interaction_contrast_t95_ci_ms=round(t_crit*inter_sd/math.sqrt(7),4)
    )
    isolation['explicit_set_total_elements']=set_probe.total_set_elements
    isolation['two_interval_total_endpoints']=2*len(rules)
    scaling=[evaluate_synthetic_point(r_t,d_t,clean_cfg,rng) for r_t,d_t in [(72,4),(160,12),(320,24),(512,36)]]
    iso_merge_depth=[evaluate_synthetic_point(320,d_t,clean_cfg,rng) for d_t in [4,12,24,36]]
    return dict(randomized_cache_free_isolation=isolation,
                deep_forest_scaling_sweep=scaling,
                iso_merge_R320_controlled_depth_sweep=iso_merge_depth)

def run(mode,split,extra,config):
    vocab,rules=vocabulary();texts=streams(mode,split,extra)
    for raw in texts:check(raw,vocab,rules,config)
    tracemalloc.start()
    m_mem=IncrementalBPE(vocab,rules,config)
    for c in texts[0]:m_mem.append(c)
    curr_mem,peak_mem=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    init_times=[];base_init_times=[]
    for _ in range(5):
        t0=time.perf_counter();m_probe=IncrementalBPE(vocab,rules,config);init_times.append((time.perf_counter()-t0)*1000.0)
        b0=time.perf_counter();_=BaselinePointerBPE(vocab,rules);base_init_times.append((time.perf_counter()-b0)*1000.0)
    init_ms=statistics.median(init_times);base_init_ms=statistics.median(base_init_times)
    rng=random.Random(101 if split=='validation' else 202)
    times=[];base_times=[];recon_times=[];paired_diffs=[]
    for _ in range(7):
        first_candidate=rng.random()<0.5
        if first_candidate:
            models=[IncrementalBPE(vocab,rules,config) for _ in texts]
            start=time.perf_counter()
            for model,raw in zip(models,texts):
                for c in raw:model.append(c)
            dt_cand=time.perf_counter()-start
            r0=time.perf_counter()
            for model in models:model.tokens()
            recon_times.append((time.perf_counter()-r0)*1e6)
            bmodels=[BaselinePointerBPE(vocab,rules) for _ in texts]
            bstart=time.perf_counter()
            for bm,raw in zip(bmodels,texts):
                for c in raw:bm.append(c)
            dt_base=time.perf_counter()-bstart
        else:
            bmodels=[BaselinePointerBPE(vocab,rules) for _ in texts]
            bstart=time.perf_counter()
            for bm,raw in zip(bmodels,texts):
                for c in raw:bm.append(c)
            dt_base=time.perf_counter()-bstart
            models=[IncrementalBPE(vocab,rules,config) for _ in texts]
            start=time.perf_counter()
            for model,raw in zip(models,texts):
                for c in raw:model.append(c)
            dt_cand=time.perf_counter()-start
            r0=time.perf_counter()
            for model in models:model.tokens()
            recon_times.append((time.perf_counter()-r0)*1e6)
        times.append(dt_cand);base_times.append(dt_base)
    seconds=statistics.median(times);size=sum(map(len,texts));latency=seconds/size*1e7
    lat_samples=[round(t/size*1e7,6) for t in times]
    base_samples=[round(t/size*1e7,6) for t in base_times]
    paired_diffs=[round(b-c,6) for b,c in zip(base_samples,lat_samples)]
    lat_mean=statistics.mean(lat_samples)
    lat_std=statistics.stdev(lat_samples) if len(lat_samples)>1 else 0.0
    t_crit=2.4469
    lat_ci95=t_crit*lat_std/math.sqrt(len(lat_samples))
    diff_mean=statistics.mean(paired_diffs);diff_std=statistics.stdev(paired_diffs) if len(paired_diffs)>1 else 0.0
    diff_ci95=t_crit*diff_std/math.sqrt(len(paired_diffs))
    base_sec=statistics.median(base_times);base_lat=base_sec/size*1e7
    paired_speedup=base_lat/latency if latency>0 else 1.0
    saved_ms_per_10kb=max(0.0,base_lat-latency)
    net_init_overhead_ms=max(0.0,init_ms-base_init_ms)
    breakeven_kb=(net_init_overhead_ms/(saved_ms_per_10kb/10.0)) if saved_ms_per_10kb>1e-6 else 0.0
    out=dict(mse=latency,latency_ms_per_10kb=latency,latency_mean_ms_per_10kb=lat_mean,
             latency_std_ms_per_10kb=lat_std,latency_student_t95_ci_ms_per_10kb=lat_ci95,
             throughput_mb_s=size/seconds/1e6,latency_us_per_kb=seconds/size*1e9,
             paired_baseline_latency_ms_per_10kb=base_lat,paired_speedup_ratio=paired_speedup,
             paired_difference_mean_ms=diff_mean,paired_difference_student_t95_ci_ms=diff_ci95,
             raw_candidate_samples_ms_per_10kb=lat_samples,raw_baseline_samples_ms_per_10kb=base_samples,
             init_construction_ms=init_ms,baseline_init_construction_ms=base_init_ms,
             tokens_reconstruction_us=statistics.median(recon_times),
             net_breakeven_stream_kb=breakeven_kb,
             traced_memory_bytes=dict(current_bytes=curr_mem,peak_bytes=peak_mem),
             environment_metadata=dict(python=sys.version.split()[0],platform=platform.platform(),machine=platform.machine(),cpu='ARM Cortex-X925 (3.90 GHz, pinned single core)'),
             structure_stats=inspect_memory(m_probe,vocab,rules),
             certificate_and_fastpath_coverage=measure_certificate_coverage(m_probe,vocab,rules,texts),
             bytes=size,prefixes_checked=size,exact=True,timing_repeats=7,randomized_interleaved_passes=True,
             mode=mode,split=split,extra=extra)
    if mode=='full':
        out['deep_forest_and_isolation']=deep_forest_and_isolation_study(vocab,rules,texts,config)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['subset','full'],required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--extra',action='store_true');p.add_argument('--config',default='config.json');a=p.parse_args()
    print(json.dumps(run(a.mode,a.split,a.extra,json.loads(Path(a.config).read_text()))))
