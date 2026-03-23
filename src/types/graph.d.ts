export interface Node {
  id: string;
  name?: string;
  val?: number;
  color?: string;
  [key: string]: string | number | boolean | undefined;
}

export interface Link {
  source: string;
  target: string;
  color?: string;
  value?: number;
  [key: string]: string | number | boolean | undefined;
}

export interface GraphData {
  nodes: Node[];
  links: Link[];
}
