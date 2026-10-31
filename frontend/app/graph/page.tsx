'use client';

import { useState, useEffect, useCallback } from 'react';
import ReactFlow, { 
  Node, 
  Edge, 
  NodeMouseHandler,
  Controls, 
  Background, 
  useNodesState, 
  useEdgesState 
} from 'reactflow';
import 'reactflow/dist/style.css';
import { apiClient } from '@/lib/api';
import { GraphResponse } from '@/types/api';

interface GraphNodeData {
  label: string;
  type: string;
}

export default function GraphPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState<GraphNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<Node<GraphNodeData> | null>(null);

  const fetchGraph = useCallback(async () => {
    try {
      const data = (await apiClient.getGraph()) as GraphResponse;
      const flowNodes: Node<GraphNodeData>[] = data.nodes.map((n, i) => ({
        id: n.id,
        data: { label: n.label, type: n.type },
        position: { x: i * 150, y: i * 100 },
      }));
      const flowEdges: Edge[] = data.edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.label,
      }));
      setNodes(flowNodes);
      setEdges(flowEdges);
    } catch (err) {
      console.error(err);
      setNodes([]);
      setEdges([]);
    } finally {
      setLoading(false);
    }
  }, [setNodes, setEdges]);

  useEffect(() => {
    void fetchGraph();
  }, [fetchGraph]);

  const onNodeClick: NodeMouseHandler = useCallback((_event, node) => {
    setSelectedNode(node as Node<GraphNodeData>);
  }, []);

  return (
    <div className="flex h-full">
      <div className="flex-1">
        <div className="p-4 border-b border-gray-200 bg-white">
          <h1 className="text-xl font-semibold text-gray-900">Graph Explorer</h1>
        </div>
        <div className="h-[calc(100%-64px)]">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500">Loading...</p>
            </div>
          ) : nodes.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500">No graph data available</p>
            </div>
          ) : (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              fitView
            >
              <Controls />
              <Background />
            </ReactFlow>
          )}
        </div>
      </div>
      {selectedNode && (
        <div className="w-80 border-l border-gray-200 bg-white p-4">
          <h2 className="font-medium text-gray-900 mb-2">Node Details</h2>
          <p className="text-sm text-gray-600 mb-1">Label: {selectedNode.data.label}</p>
          <p className="text-sm text-gray-600 mb-1">Type: {selectedNode.data.type}</p>
          <p className="text-xs text-gray-500">ID: {selectedNode.id}</p>
        </div>
      )}
    </div>
  );
}
