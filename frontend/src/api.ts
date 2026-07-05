import type {
  CreateGameRequest,
  CreateGameResponse,
  GameDetailResponse,
  LegalActionsResponse,
  MoveDTO,
  PlayMoveResponse,
} from "./types/api";

const API_BASE = "";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new ApiError(resp.status, body?.detail?.error?.code ?? "UNKNOWN", body?.detail?.error?.message ?? resp.statusText);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json() as Promise<T>;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

export async function createGame(req: CreateGameRequest): Promise<CreateGameResponse> {
  return request<CreateGameResponse>("/api/v1/games", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function getGame(gameId: string, sessionToken: string): Promise<GameDetailResponse> {
  return request<GameDetailResponse>(`/api/v1/games/${gameId}`, {
    headers: { "X-Quoridor-Session": sessionToken },
  });
}

export async function playMove(
  gameId: string,
  sessionToken: string,
  action: MoveDTO,
): Promise<PlayMoveResponse> {
  return request<PlayMoveResponse>(`/api/v1/games/${gameId}/moves`, {
    method: "POST",
    headers: { "X-Quoridor-Session": sessionToken },
    body: JSON.stringify({ action }),
  });
}

export async function getLegalActions(
  gameId: string,
  sessionToken: string,
): Promise<LegalActionsResponse> {
  return request<LegalActionsResponse>(`/api/v1/games/${gameId}/legal-actions`, {
    headers: { "X-Quoridor-Session": sessionToken },
  });
}
