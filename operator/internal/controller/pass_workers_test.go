/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	v1alpha1 "github.com/NVIDIA/nodewright/operator/api/nodewright/v1alpha1"
	skyhookv1 "github.com/NVIDIA/nodewright/operator/api/v1alpha1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

var _ = Describe("forEachNode", func() {
	DescribeTable("visits every index exactly once", func(workers int) {
		var calls sync.Map
		visited := forEachNode(workers, 200, func(i int) bool {
			_, dup := calls.LoadOrStore(i, true)
			Expect(dup).To(BeFalse())
			return false
		})
		for i, v := range visited {
			Expect(v).To(BeTrue())
			_, ok := calls.Load(i)
			Expect(ok).To(BeTrue())
		}
	}, Entry("one at a time", 1), Entry("zero means one at a time", 0), Entry("sixteen workers", 16))

	It("stops at the first stop when running one at a time", func() {
		visited := forEachNode(1, 50, func(i int) bool { return i == 7 })
		for i, v := range visited {
			Expect(v).To(Equal(i <= 7))
		}
	})

	It("dispatches nothing new after a stop and runs every node it dispatched", func() {
		var ran [100]atomic.Bool
		visited := forEachNode(4, 100, func(i int) bool {
			ran[i].Store(true)
			time.Sleep(time.Millisecond)
			return i == 10
		})
		count := 0
		for i, v := range visited {
			Expect(ran[i].Load()).To(Equal(v))
			if v {
				count++
			}
		}
		Expect(visited[10]).To(BeTrue())
		Expect(count).To(BeNumerically("<", 100))
	})
})

func sortedKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

var _ = Describe("PASS_WRITE_WORKERS rollout equivalence", func() {
	const namespace = "nodewright"

	newNode := func(i int) *corev1.Node {
		return &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("worker-%02d", i), Labels: map[string]string{"pool": "gpu"}},
			Status: corev1.NodeStatus{
				Conditions: []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionTrue}},
				NodeInfo:   corev1.NodeSystemInfo{BootID: fmt.Sprintf("boot-%02d", i)},
			},
		}
	}

	nodeWrights := func() []*v1alpha1.NodeWright {
		budget := 6
		selector := metav1.LabelSelector{MatchLabels: map[string]string{"pool": "gpu"}}
		first := &v1alpha1.NodeWright{
			ObjectMeta: metav1.ObjectMeta{Name: "first", UID: "first", Generation: 1},
			Spec: v1alpha1.NodeWrightSpec{
				Priority:           1,
				NodeSelector:       selector,
				InterruptionBudget: v1alpha1.InterruptionBudget{Count: &budget},
				Packages: v1alpha1.Packages{
					"tune": {PackageRef: v1alpha1.PackageRef{Name: "tune", Version: "1.0.0"}, Image: "example/tune"},
					"bounce": {
						PackageRef: v1alpha1.PackageRef{Name: "bounce", Version: "1.0.0"}, Image: "example/bounce",
						Interrupt: &v1alpha1.Interrupt{Type: v1alpha1.NOOP},
					},
				},
			},
		}
		second := &v1alpha1.NodeWright{
			ObjectMeta: metav1.ObjectMeta{Name: "second", UID: "second", Generation: 1},
			Spec: v1alpha1.NodeWrightSpec{
				Priority:           2,
				Serial:             true,
				NodeSelector:       selector,
				InterruptionBudget: v1alpha1.InterruptionBudget{Count: &budget},
				Packages: v1alpha1.Packages{
					"alpha": {PackageRef: v1alpha1.PackageRef{Name: "alpha", Version: "1.0.0"}, Image: "example/alpha"},
					"beta":  {PackageRef: v1alpha1.PackageRef{Name: "beta", Version: "1.0.0"}, Image: "example/beta"},
				},
			},
		}
		return []*v1alpha1.NodeWright{first, second}
	}

	type rollout struct {
		nodes      map[string]string
		nodeWright map[string]string
		jobs       []string
		passes     int
		complete   bool
	}

	opts := func(workers int) SkyhookOperatorOptions {
		return SkyhookOperatorOptions{
			Namespace: namespace, CopyDirRoot: "/var/lib/skyhook", AgentLogRoot: "/var/log/skyhook",
			RuntimeRequiredTaint: "nodewright.nvidia.com=runtime-required:NoSchedule",
			AgentImage:           "ghcr.io/nvidia/nodewright/agent:1.2.3", PauseImage: "registry.k8s.io/pause:3.10",
			MaxInterval:        10 * time.Minute,
			PassWriteWorkers:   workers,
			JobOperatorOptions: JobOperatorOptions{JobTTLSucceeded: time.Hour, JobTTLFailed: 24 * time.Hour, JobStageTimeout: time.Hour, JobBackoffLimit: 3},
		}
	}

	snapshot := func(ctx context.Context, c client.Client) (map[string]string, map[string]string) {
		nodes := &corev1.NodeList{}
		Expect(c.List(ctx, nodes)).To(Succeed())
		nodeView := map[string]string{}
		for _, n := range nodes.Items {
			keys := []string{}
			for k, v := range n.Annotations {
				if strings.Contains(k, "/nodeState_") || strings.Contains(k, "/status_") {
					keys = append(keys, "a:"+k+"="+v)
				}
			}
			for k, v := range n.Labels {
				if strings.Contains(k, "/status_") {
					keys = append(keys, "l:"+k+"="+v)
				}
			}
			sort.Strings(keys)
			nodeView[n.Name] = fmt.Sprintf("unschedulable=%t %s", n.Spec.Unschedulable, strings.Join(keys, " "))
		}
		nws := &v1alpha1.NodeWrightList{}
		Expect(c.List(ctx, nws)).To(Succeed())
		nwView := map[string]string{}
		for _, nw := range nws.Items {
			nwView[nw.Name] = fmt.Sprintf("status=%s nodeStatus=%v nodeState=%v priority=%d offset=%d",
				nw.Status.Status, nw.Status.NodeStatus, nw.Status.NodeState, len(nw.Status.NodePriority), nw.Status.NodeOrderOffset)
		}
		return nodeView, nwView
	}

	run := func(workers int, nodeCount int, failWrite func(obj client.Object) bool) rollout {
		ctx := context.Background()
		scheme := runtime.NewScheme()
		Expect(corev1.AddToScheme(scheme)).To(Succeed())
		Expect(batchv1.AddToScheme(scheme)).To(Succeed())
		Expect(v1alpha1.AddToScheme(scheme)).To(Succeed())
		Expect(skyhookv1.AddToScheme(scheme)).To(Succeed())

		objects := []client.Object{}
		for _, nw := range nodeWrights() {
			objects = append(objects, nw)
		}
		for i := range nodeCount {
			objects = append(objects, newNode(i))
		}

		var rngMu sync.Mutex
		rng := rand.New(rand.NewPCG(uint64(workers), 42))
		delay := func() {
			rngMu.Lock()
			d := time.Duration(rng.IntN(200)) * time.Microsecond
			rngMu.Unlock()
			time.Sleep(d)
		}
		check := func(obj client.Object) error {
			if failWrite != nil && failWrite(obj) {
				return errors.New("injected write failure")
			}
			return nil
		}
		c := newFakeClientBuilder().WithScheme(scheme).WithObjects(objects...).
			WithStatusSubresource(&v1alpha1.NodeWright{}, &batchv1.Job{}).
			WithIndex(&corev1.Pod{}, fieldSelectorNodeName, indexPodByNodeName).
			WithInterceptorFuncs(interceptor.Funcs{
				Create: func(ctx context.Context, cl client.WithWatch, obj client.Object, o ...client.CreateOption) error {
					delay()
					if err := check(obj); err != nil {
						return err
					}
					return cl.Create(ctx, obj, o...)
				},
				Patch: func(ctx context.Context, cl client.WithWatch, obj client.Object, p client.Patch, o ...client.PatchOption) error {
					delay()
					if err := check(obj); err != nil {
						return err
					}
					return cl.Patch(ctx, obj, p, o...)
				},
				Update: func(ctx context.Context, cl client.WithWatch, obj client.Object, o ...client.UpdateOption) error {
					delay()
					return cl.Update(ctx, obj, o...)
				},
				SubResourcePatch: func(ctx context.Context, cl client.Client, sub string, obj client.Object, p client.Patch, o ...client.SubResourcePatchOption) error {
					delay()
					if _, isNode := obj.(*corev1.Node); isNode {
						raw, err := p.Data(obj)
						if err != nil {
							return err
						}
						if err := cl.Patch(ctx, obj.DeepCopyObject().(client.Object), client.RawPatch(p.Type(), raw)); err != nil {
							return err
						}
						return cl.SubResource(sub).Patch(ctx, obj, client.RawPatch(p.Type(), raw), o...)
					}
					return cl.SubResource(sub).Patch(ctx, obj, p, o...)
				},
			}).Build()

		r, err := NewSkyhookReconciler(scheme, c, c, k8sfake.NewClientset(), &events.FakeRecorder{}, opts(workers))
		Expect(err).NotTo(HaveOccurred())
		jr := NewJobReconciler(c, c, k8sfake.NewClientset(), &events.FakeRecorder{}, opts(workers).JobOperatorOptions)

		seen := map[string]bool{}
		out := rollout{}
		for out.passes = 1; out.passes <= 120; out.passes++ {
			_, err := r.Reconcile(ctx, reconcile.Request{NamespacedName: types.NamespacedName{Name: "nodewright"}})
			if failWrite == nil {
				Expect(err).NotTo(HaveOccurred())
			}

			jobs := &batchv1.JobList{}
			Expect(c.List(ctx, jobs, client.InNamespace(namespace))).To(Succeed())
			finished := map[string]bool{}
			for i := range jobs.Items {
				job := &jobs.Items[i]
				seen[job.Name] = true
				if hasJobCondition(job, batchv1.JobComplete) {
					continue
				}
				now := metav1.Now()
				job.Status.StartTime = &now
				job.Status.CompletionTime = &now
				job.Status.Succeeded = 1
				job.Status.Conditions = append(job.Status.Conditions,
					batchv1.JobCondition{Type: batchv1.JobSuccessCriteriaMet, Status: corev1.ConditionTrue},
					batchv1.JobCondition{Type: batchv1.JobComplete, Status: corev1.ConditionTrue})
				Expect(c.Status().Update(ctx, job)).To(Succeed())
				finished[job.Spec.Template.Spec.NodeName] = true
			}
			for _, nodeName := range sortedKeys(finished) {
				_, err := jr.Reconcile(ctx, reconcile.Request{NamespacedName: types.NamespacedName{Namespace: namespace, Name: nodeName}})
				Expect(err).NotTo(HaveOccurred())
			}

			nws := &v1alpha1.NodeWrightList{}
			Expect(c.List(ctx, nws)).To(Succeed())
			done := true
			for _, nw := range nws.Items {
				done = done && nw.Status.Status == v1alpha1.StatusComplete
			}
			if done {
				out.complete = true
				break
			}
		}
		out.nodes, out.nodeWright = snapshot(ctx, c)
		for name := range seen {
			out.jobs = append(out.jobs, name)
		}
		sort.Strings(out.jobs)
		return out
	}

	It("ends in the same state with one worker and with sixteen", func() {
		serial := run(1, 12, nil)
		Expect(serial.complete).To(BeTrue(), "the one-worker rollout did not finish in %d passes", serial.passes)

		parallel := run(16, 12, nil)
		Expect(parallel.complete).To(BeTrue(), "the sixteen-worker rollout did not finish in %d passes", parallel.passes)

		Expect(parallel.nodes).To(Equal(serial.nodes))
		Expect(parallel.nodeWright).To(Equal(serial.nodeWright))
		Expect(parallel.jobs).To(Equal(serial.jobs))
	})

	It("keeps going on other nodes and still saves after a node's write fails", func() {
		var failures atomic.Int32
		failing := func(obj client.Object) bool {
			job, ok := obj.(*batchv1.Job)
			if ok && job.Spec.Template.Spec.NodeName == "worker-07" && failures.Load() < 3 {
				failures.Add(1)
				return true
			}
			return false
		}
		out := run(16, 12, failing)
		Expect(failures.Load()).To(BeNumerically(">", 0))
		Expect(out.complete).To(BeTrue(), "the rollout did not recover from the injected failures in %d passes", out.passes)
	})
})
